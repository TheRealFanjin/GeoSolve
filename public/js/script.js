const fileInput = document.getElementById('file-upload');
const uploadForm = document.getElementById('upload-form');
const secondScreen = document.getElementById("second-screen");
const inputImageScreen = document.getElementById("input-image-screen");
document.body.classList.add("no-scroll");

window.onbeforeunload = () => {
    window.scrollTo(0,0);
};

const submitForm = async () => {
    if(fileInput.files.length > 0) {
        const file = fileInput.files[0];

        const formData = new FormData();
        formData.append("inputImage", file);

        try {
            const response = await fetch("/upload", {
                method: "POST",
                body: formData
            });

            if(response.ok) {
                secondScreen.removeAttribute("hidden");
                inputImageScreen.setAttribute("src", URL.createObjectURL(file))
                secondScreen.scrollIntoView({
                    behavior: "smooth",
                    block: "center"
                });
                uploadForm.reset();
            } else {
                const errorData = await response.json();
                alert(`Upload failed: ${errorData.message || 'Unknown error'}`);
            }
        }
        catch (error) {
            alert("An error occurred during upload. Please try again.");
        }
    }
}



fileInput.addEventListener('change', submitForm);


