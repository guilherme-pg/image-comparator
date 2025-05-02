
// Get the form and the submit button
const submitButton = document.getElementById('upload-button');

// Add an event listener to the form
submitButton.addEventListener('click', function(event) {
    // Check if either image input is empty
    const input1 = document.getElementById('image_input_1');
    const input2 = document.getElementById('image_input_2');

    if (!input1.files.length || !input2.files.length) {
        event.preventDefault();
        alert('Please select both images.');
    }
});



function setupDropArea(dropAreaId, inputId) {
    const dropArea = document.getElementById(dropAreaId);
    const input = document.getElementById(inputId);
    const label = dropArea.querySelector('.file-label');

    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];

    function showFileName(file) {
        if (file) {
            label.textContent = file.name;
        }
    }

    function isValidFile(file) {
        return allowedTypes.includes(file.type);
    }

    function handleFile(file) {
        if (isValidFile(file)) {
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            input.files = dataTransfer.files;
            showFileName(file);
        } else {
            label.textContent = 'Invalid file type. Only PNG, JPG or JPEG are allowed.';
            input.value = ''; // limpa o input
        }
    }

    // Drag and drop
    dropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropArea.classList.add('dragover');
    });

    dropArea.addEventListener('dragleave', () => {
        dropArea.classList.remove('dragover');
    });

    dropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        dropArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Manual selection
    input.addEventListener('change', () => {
        if (input.files.length > 0) {
            handleFile(input.files[0]);
        }
    });
}

document.addEventListener('DOMContentLoaded', () => {
    setupDropArea('drop-area-1', 'image_input_1');
    setupDropArea('drop-area-2', 'image_input_2');
});