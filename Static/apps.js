const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const previewPlaceholder = document.getElementById("previewPlaceholder");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const errorBox = document.getElementById("errorBox");

const dateEnteredField = document.getElementById("date_entered");
const descriptionField = document.getElementById("description");
const nameField = document.getElementById("name");
const brandField = document.getElementById("brand_make");
const modelField = document.getElementById("model");
const typeField = document.getElementById("type");
const specsField = document.getElementById("visible_specs");
const markingsField = document.getElementById("visible_markings");
const summaryField = document.getElementById("enriched_summary");
const sourceUrlField = document.getElementById("source_url");
const confidenceField = document.getElementById("confidence_notes");
const rawJsonField = document.getElementById("raw_json");

function todayLocalISO() {
  const now = new Date();
  const tzOffset = now.getTimezoneOffset() * 60000;
  return new Date(now - tzOffset).toISOString().split("T")[0];
}

function setDefaultDate(value = null) {
  dateEnteredField.value = value || todayLocalISO();
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.style.display = "block";
}

function clearError() {
  errorBox.textContent = "";
  errorBox.style.display = "none";
}

function clearResults() {
  setDefaultDate();
  descriptionField.value = "";
  nameField.value = "";
  brandField.value = "";
  modelField.value = "";
  typeField.value = "";
  specsField.value = "";
  markingsField.value = "";
  summaryField.value = "";
  sourceUrlField.value = "";
  confidenceField.value = "";
  rawJsonField.textContent = "";
}

function clearAll() {
  imageInput.value = "";
  previewImage.src = "";
  previewImage.style.display = "none";
  previewPlaceholder.style.display = "inline";
  clearError();
  clearResults();
}

imageInput.addEventListener("change", () => {
  clearError();

  const file = imageInput.files[0];
  if (!file) {
    previewImage.src = "";
    previewImage.style.display = "none";
    previewPlaceholder.style.display = "inline";
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    previewImage.style.display = "block";
    previewPlaceholder.style.display = "none";
  };
  reader.readAsDataURL(file);
});

clearBtn.addEventListener("click", clearAll);

analyzeBtn.addEventListener("click", async () => {
  clearError();

  const file = imageInput.files[0];
  if (!file) {
    showError("Please select an image first.");
    return;
  }

  descriptionField.value = "";
  nameField.value = "";
  brandField.value = "";
  modelField.value = "";
  typeField.value = "";
  specsField.value = "";
  markingsField.value = "";
  summaryField.value = "";
  sourceUrlField.value = "";
  confidenceField.value = "";
  rawJsonField.textContent = "";

  const formData = new FormData();
  formData.append("image", file);

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      body: formData
    });

    const rawText = await response.text();

    if (!rawText || !rawText.trim()) {
      throw new Error("Server returned an empty response.");
    }

    let data;
    try {
      data = JSON.parse(rawText);
    } catch (err) {
      console.error("Invalid JSON:", rawText);
      throw new Error("Server did not return valid JSON.");
    }

    if (!response.ok || !data.success) {
      throw new Error(data.error || "Request failed.");
    }

    setDefaultDate(data.date_entered);

    const result = data.result || {};
    descriptionField.value = result.description || "";
    nameField.value = result.name || "";
    brandField.value = result.brand_make || "";
    modelField.value = result.model || "";
    typeField.value = result.type || "";
    specsField.value = result.visible_specs || "";
    markingsField.value = result.visible_markings || "";
    summaryField.value = result.enriched_summary || "";
    sourceUrlField.value = result.source_url || "";
    confidenceField.value = result.confidence_notes || "";
    rawJsonField.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    console.error(err);
    showError(err.message || "Something went wrong.");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze";
  }
});

document.addEventListener("DOMContentLoaded", () => {
  setDefaultDate();
});
