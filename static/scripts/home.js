
// Get the form and the submit button
const submitButton = document.getElementById('upload-button');

// Add an event listener to the form
submitButton.addEventListener('click', function(event) {
    // Check if either image input is empty
    const image1 = document.getElementById('image_input_1').value;
    const image2 = document.getElementById('image_input_2').value;

    if (!image1 || !image2) {
        event.preventDefault(); // Prevent the form submission
        alert('Please select both images.'); // Display an alert message
    }
});