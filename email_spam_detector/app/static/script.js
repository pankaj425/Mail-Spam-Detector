const messageInput = document.getElementById("messageInput");
const predictBtn = document.getElementById("predictBtn");
const exampleBtn = document.getElementById("exampleBtn");
const resultText = document.getElementById("resultText");

const trainFileInput = document.getElementById("trainFile");
const trainFileBtn = document.getElementById("trainFileBtn");
const trainDemoBtn = document.getElementById("trainDemoBtn");
const downloadModelBtn = document.getElementById("downloadModelBtn");
const trainStatus = document.getElementById("trainStatus");

exampleBtn.addEventListener("click", () => {
  messageInput.value =
    "Congratulations! You have won a lottery of ₹10,00,000. Claim your prize now!";
  resultText.textContent = "";
});

predictBtn.addEventListener("click", async () => {
  const msg = messageInput.value.trim();
  if (!msg) {
    resultText.textContent = "Please enter a message first.";
    return;
  }

  resultText.textContent = "Predicting...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg }),
    });

    const data = await response.json();

    if (!response.ok) {
      resultText.textContent = data.error || "Something went wrong.";
      return;
    }

    resultText.textContent = "Prediction: " + data.prediction;
  } catch (err) {
    console.error(err);
    resultText.textContent = "Network error. Check if Flask is running.";
  }
});

trainDemoBtn.addEventListener("click", async () => {
  trainStatus.textContent = "Training on demo data...";
  try {
    const response = await fetch("/train-demo", { method: "POST" });
    const data = await response.json();
    if (!response.ok) {
      trainStatus.textContent = data.error || "Training failed.";
      return;
    }
    trainStatus.textContent =
      "Demo training complete. Samples used: " + data.samples;
  } catch (err) {
    console.error(err);
    trainStatus.textContent = "Network error while training.";
  }
});

trainFileBtn.addEventListener("click", async () => {
  const file = trainFileInput.files[0];
  if (!file) {
    trainStatus.textContent = "Please choose a CSV file first.";
    return;
  }

  trainStatus.textContent = "Uploading and training on file...";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/train-file", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      trainStatus.textContent = data.error || "Training failed.";
      return;
    }
    trainStatus.textContent =
      "Training on file complete. Samples used: " + data.samples;
  } catch (err) {
    console.error(err);
    trainStatus.textContent = "Network error while training.";
  }
});

downloadModelBtn.addEventListener("click", () => {
  window.location.href = "/download-model";
});
