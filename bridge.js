
const express = require('express');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const bodyParser = require('body-parser');

const app = express();
const PORT = process.env.PORT || 4000; 

// --- CONFIGURATION FOR RUNPOD ---

// Fallback VPS Configuration (Used if templateUrl is missing)
const DEFAULT_VPS_HOST = 'http://185.202.223.229:3001'; 

// RunPod Paths (Using /workspace for persistence)
const WORKSPACE_ROOT = '/workspace';
const FACEFUSION_ROOT = path.join(WORKSPACE_ROOT, 'facefusion');
const BRIDGE_ROOT = path.join(WORKSPACE_ROOT, 'bridge_data');

// Sub-directories
const TEMPLATES_DIR = path.join(BRIDGE_ROOT, 'templates'); // Cached templates live here
const INPUT_DIR = path.join(BRIDGE_ROOT, 'inputs');        // Temporary user selfies
const OUTPUT_DIR = path.join(BRIDGE_ROOT, 'outputs');      // Result videos

// Ensure directories exist
[TEMPLATES_DIR, INPUT_DIR, OUTPUT_DIR].forEach(dir => {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
});

// FaceFusion Configuration
const PROVIDER = 'cuda'; // RunPod uses NVIDIA GPUs
const OUTPUT_QUALITY = '80';

app.use(bodyParser.json({ limit: '200mb' }));

// --- HELPER: DOWNLOAD FILE ---
const downloadFile = async (url, destPath) => {
    console.log(`[Cache] Downloading: ${url} -> ${destPath}`);
    const writer = fs.createWriteStream(destPath);
    
    try {
        const response = await axios({
            url,
            method: 'GET',
            responseType: 'stream',
            timeout: 60000 // 60s timeout for large files
        });

        response.data.pipe(writer);

        return new Promise((resolve, reject) => {
            writer.on('finish', resolve);
            writer.on('error', reject);
        });
    } catch (e) {
        if (fs.existsSync(destPath)) fs.unlinkSync(destPath); // Delete partial file
        throw new Error(`Download failed: ${e.message}`);
    }
};

// --- HEALTH CHECK ---
app.get('/ping', (req, res) => {
    res.json({ 
        status: 'online', 
        platform: 'RunPod Optimized',
        dirs: { templates: TEMPLATES_DIR, output: OUTPUT_DIR }
    });
});

// --- PROCESS TASK ---
app.post('/process', async (req, res) => {
    const { requestId, templateId, faceBase64, callbackUrl, userId, templateUrl } = req.body;
    
    console.log(`\n========================================`);
    console.log(`[Bridge] 🎬 NEW TASK: ${requestId}`);
    console.log(`[Bridge] Template ID: ${templateId}`);
    
    // 1. Respond immediately to VPS (Async processing)
    res.json({ status: 'processing', requestId }); 

    // Define Paths
    const templateFileName = `${templateId}.mp4`; 
    const localTemplatePath = path.join(TEMPLATES_DIR, templateFileName);
    const faceFilePath = path.join(INPUT_DIR, `${requestId}_face.jpg`);
    const outputVideoPath = path.join(OUTPUT_DIR, `${requestId}_result.mp4`);

    // Determine download URL: Use provided one or fallback
    const downloadUrl = templateUrl || `${DEFAULT_VPS_HOST}/templates/${templateFileName}`;

    try {
        // 2. CHECK CACHE / DOWNLOAD TEMPLATE
        // This is the optimization: We only download if we don't have it.
        if (!fs.existsSync(localTemplatePath)) {
            console.log(`[Cache] Template miss. Downloading from source...`);
            console.log(`[Cache] Source URL: ${downloadUrl}`);
            await downloadFile(downloadUrl, localTemplatePath);
            console.log(`[Cache] Template downloaded and cached.`);
        } else {
            console.log(`[Cache] Template hit! Using local copy.`);
        }

        // 3. Save User Face
        const base64Data = faceBase64.replace(/^data:image\/\w+;base64,/, "");
        fs.writeFileSync(faceFilePath, Buffer.from(base64Data, 'base64'));

        // 4. Run FaceFusion
        console.log(`[Engine] Starting FaceFusion...`);
        
        // Command construction
        const args = [
            'facefusion.py',
            'headless-run',
            '--source-paths', faceFilePath,
            '--target-path', localTemplatePath,
            '--output-path', outputVideoPath,
            '--processors', 'face_swapper', 'face_enhancer',
            '--execution-providers', PROVIDER,
            '--output-video-preset', 'ultrafast', // Speed optimization
            '--output-video-quality', OUTPUT_QUALITY,
            '--keep-fps'
        ];

        const child = spawn('python3', args, { cwd: FACEFUSION_ROOT });

        let logs = "";
        child.stdout.on('data', (d) => { process.stdout.write(d); logs += d.toString(); });
        child.stderr.on('data', (d) => { process.stderr.write(d); logs += d.toString(); });

        child.on('close', async (code) => {
            if (code === 0 && fs.existsSync(outputVideoPath)) {
                console.log(`[Engine] ✅ Render Complete! Sending to VPS...`);
                
                const videoBuffer = fs.readFileSync(outputVideoPath);
                
                // Send back to VPS
                await axios.post(callbackUrl, {
                    requestId, userId, success: true,
                    videoBase64: videoBuffer.toString('base64')
                }, { maxContentLength: Infinity, maxBodyLength: Infinity })
                .then(() => console.log(`[Bridge] 📡 Uploaded result to VPS.`))
                .catch(e => console.error(`[Bridge] Callback Error: ${e.message}`));

            } else {
                console.error(`[Engine] ❌ FAILED (Code ${code})`);
                await axios.post(callbackUrl, { 
                    requestId, userId, success: false, 
                    error: `Render failed. Exit code ${code}` 
                }).catch(() => {});
            }

            // 5. Cleanup (Delete temp face and output, KEEP TEMPLATE)
            try {
                if(fs.existsSync(faceFilePath)) fs.unlinkSync(faceFilePath);
                if(fs.existsSync(outputVideoPath)) fs.unlinkSync(outputVideoPath);
            } catch(e) {}
        });

    } catch (err) {
        console.error("[Bridge Fatal Error]", err);
        await axios.post(callbackUrl, { requestId, userId, success: false, error: err.message }).catch(() => {});
    }
});

// Start Server
app.listen(PORT, () => {
    console.log(`\n🚀 RunPod Bridge Started on port ${PORT}`);
    console.log(`📂 Cache Dir: ${TEMPLATES_DIR}`);
    console.log(`🔗 Default VPS Host:  ${DEFAULT_VPS_HOST}`);
});
