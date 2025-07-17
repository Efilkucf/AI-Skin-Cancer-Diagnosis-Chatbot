const imageInput = document.getElementById("imageInput");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const chatBox = document.getElementById("chatBox");

const backendUrl = "https://ai-skin-cancer-diagnosis-chatbot-1.onrender.com";

let uploadedImage = null;

imageInput.addEventListener("change", () => {
  uploadedImage = imageInput.files[0];
  if (!uploadedImage) return;

  const reader = new FileReader();
  reader.onload = function (e) {
    // Step 1: Add user bubble with image
    const imgHTML = `<img src="${e.target.result}" class="preview-image" alt="Uploaded Image" />`;
    appendMessage("user", imgHTML);

    // Step 2: Add placeholder bot response ("Analyzing...")
    const botMsg = appendMessage("bot", `<em>🧠 Analyzing image...</em>`);
    const bubble = botMsg.querySelector(".bubble");

    // Step 3: Predict and update bot response
    sendToPredict(bubble);
  };
  reader.readAsDataURL(uploadedImage);
  imageInput.value = null;
});


sendBtn.addEventListener("click", () => {
  const msg = chatInput.value.trim();
  if (msg) {
    appendMessage("user", msg);
    chatInput.value = "";
    sendToAsk();
  }
});

function appendMessage(sender, content) {
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${sender}`;
  msgDiv.innerHTML = `
    <div class="bubble">${content}<span class="timestamp">${new Date().toLocaleTimeString()}</span></div>
  `;
  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
  return msgDiv;
}
function sendToPredict(targetBubble) {
  const formData = new FormData();
  formData.append("file", uploadedImage);

  fetch(`${backendUrl}/predict`, {
    method: "POST",
    body: formData,
  })
    .then(res => res.json())
    .then(data => {
      targetBubble.innerHTML = `
        <strong>Diagnosis:</strong> ${data.diagnosis}<br>
        <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%
        <div class="timestamp">${new Date().toLocaleTimeString()}</div>
      `;
    })
    .catch(() => {
      targetBubble.innerHTML = `❌ Prediction failed. Please try again.`;
    });
}


function sendToAsk() {
  const loadingMsg = appendMessage("bot", "💬 Generating answer...");

  fetch(`${backendUrl}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chat_history: getChatHistory() }),
  })
    .then(res => res.json())
    .then(data => {
      loadingMsg.querySelector(".bubble").innerHTML = `
        ${data.answer}
        <span class="timestamp">${new Date().toLocaleTimeString()}</span>
      `;
    })
    .catch(() => {
      loadingMsg.querySelector(".bubble").innerHTML = `
        Failed to get a response.
        <span class="timestamp">${new Date().toLocaleTimeString()}</span>
      `;
    });
}

function getChatHistory() {
  const messages = [];
  const bubbles = chatBox.querySelectorAll(".message");
  bubbles.forEach(bubble => {
    const role = bubble.classList.contains("user") ? "user" : "assistant";
    const text = bubble.querySelector(".bubble")?.innerText.trim();
    if (text) {
      messages.push({ role, content: text });
    }
  });
  return messages;
}
