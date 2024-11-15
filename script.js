// Fungsi untuk menangani prediksi sentimen
async function predictSentiment() {
  // Ambil teks input dan model yang dipilih
  const inputText = document.getElementById("inputText").value;
  const selectedModel = document.getElementById("modelSelect").value;

  // Cek jika input text kosong atau model tidak dipilih
  if (inputText.trim() === "") {
    alert("Please enter some text to predict the sentiment.");
    return;
  }

  if (!selectedModel) {
    alert("Please choose a model (KNN or SVM).");
    return;
  }

  // Siapkan data yang akan dikirimkan ke server
  const requestData = {
    text: inputText,
    model_type: selectedModel,
  };

  try {
    // Mengirimkan permintaan POST ke endpoint FastAPI
    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestData),
    });

    // Menangani respon dari server
    const data = await response.json();

    if (response.ok) {
      // Menampilkan hasil prediksi sentimen
      const sentiment =
        data.predicted_sentiment === 1 ? "Positive" : "Negative";

      const resultText = `
        <p><strong>Text:</strong> ${data.text}</p>
        <p><strong>Selected Model:</strong> ${selectedModel}</p>
        <p><strong>Predicted Sentiment:</strong> ${sentiment}</p>
      `;

      // Tampilkan hasil dan aktifkan kontainer hasil
      document.getElementById("resultText").innerHTML = resultText;
      document.querySelector(".result-container").style.display = "block";
    } else {
      // Menangani error jika response tidak oke
      document.getElementById("resultText").innerHTML = `
        <p><strong>Error:</strong> ${data.detail || "Something went wrong."}</p>
      `;
      document.querySelector(".result-container").style.display = "block";
    }
  } catch (error) {
    // Menangani error pada fetch atau jaringan
    document.getElementById("resultText").innerHTML = `
      <p><strong>Error:</strong> ${error.message}</p>
    `;
    document.querySelector(".result-container").style.display = "block";
  }
}
