const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const resultEl = document.getElementById("result");
const startBtn = document.getElementById("start");
const stopBtn = document.getElementById("stop");

let stream = null;
let intervalId = null;

// Start webcam
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
    video.srcObject = stream;
  } catch (err) {
    console.error("Error accessing camera:", err);
    alert("Could not access camera. Please allow camera permission in your browser.");
  }
}

// Stop webcam
function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
}

// Capture frame and send to backend
async function captureAndSendFrame() {
  if (!stream) return;

  canvas.width = video.videoWidth || 400;
  canvas.height = video.videoHeight || 300;

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const dataUrl = canvas.toDataURL("image/jpeg"); // base64 string

  try {
    const resp = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ image: dataUrl }),
    });

    const data = await resp.json();
    if (data.label) {
      resultEl.textContent = `Result: ${data.label} (confidence: ${data.confidence})`;
    } else if (data.error) {
      console.error("Server error:", data.error);
    }
  } catch (e) {
    console.error("Error sending frame:", e);
  }
}

// Buttons
startBtn.addEventListener("click", async () => {
  if (!stream) {
    await startCamera();
  }
  if (!intervalId) {
    // send a frame every 500 ms
    intervalId = setInterval(captureAndSendFrame, 500);
  }
});

stopBtn.addEventListener("click", () => {
  if (intervalId) {
    clearInterval(intervalId);
    intervalId = null;
  }
  stopCamera();
  resultEl.textContent = "Result: -";
});

// Optionally auto-start camera on page load
window.addEventListener("load", () => {
  startCamera();
});
