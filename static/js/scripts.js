function classifyImage() {
    const fileInput = document.getElementById('image-upload');
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select an image.');
        return;
    }

    const formData = new FormData();
    formData.append('image', file);

    fetch('/classify', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        displayResult(data.result);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function displayResult(result) {
    const resultDiv = document.getElementById('result');
    resultDiv.textContent = result;
}
