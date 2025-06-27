const webcam = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const resultsList = document.getElementById('results');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const ctx = canvas.getContext('2d');
let stream = null;
let processing = false;

async function startWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcam.srcObject = stream;
        webcam.play();
        canvas.width = webcam.videoWidth || 640;
        canvas.height = webcam.videoHeight || 480;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        processing = true;
        processFrames();
    } catch (err) {
        alert('Error accessing webcam: ' + err.message);
    }
}

function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        webcam.srcObject = null;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        resultsList.innerHTML = '<li class="list-group-item">No detections yet...</li>';
        startBtn.disabled = false;
        stopBtn.disabled = true;
        processing = false;
    }
}

async function processFrames() {
    if (!processing) return;
    
    ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL('image/jpeg');
    
    try {
        const response = await fetch('/process_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        
        const data = await response.json();
        if (data.error) {
            console.error('Error:', data.error);
            return;
        }
        
        const img = new Image();
        img.src = data.image;
        img.onload = () => {
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        };
        
        resultsList.innerHTML = data.detections.length > 0
            ? data.detections.map(d => `<li class="list-group-item">${d.class}: ${(d.confidence * 100).toFixed(2)}%</li>`).join('')
            : '<li class="list-group-item">No detections yet...</li>';
    } catch (err) {
        console.error('Error processing frame:', err);
    }
    
    requestAnimationFrame(processFrames);
}

startBtn.addEventListener('click', startWebcam);
stopBtn.addEventListener('click', stopWebcam);

