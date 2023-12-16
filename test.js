const container = document.querySelector('#container');
const containerVideo = document.querySelector('#container-video')
const fileInput = document.querySelector('#file-input');
const video = document.getElementById('video');

async function loadTrainingData() {
    const labels = ['YanFang', 'Fukada Eimi', 'Rina Ishihara', 'Takizawa Laura', 'Yua Mikami'];

    const faceDescriptors = [];
    for (const label of labels) {
        const descriptors = [];
        for (let i = 1; i <= label.length; i++) {
            const imageExtensions = ['jpg', 'jpeg', 'png'];

            const imagePath = await findExistingImagePath(`/data/${label}/${i}`, imageExtensions);

            if (imagePath) {
                try {
                    const image = await faceapi.fetchImage(imagePath);
                    const detection = await faceapi.detectSingleFace(image).withFaceLandmarks().withFaceDescriptor().withAgeAndGender();
                    descriptors.push(detection.descriptor);
                } catch (error) {
                    // console.error(`Không thể tải ảnh ${imagePath}.`, error);
                }
            }
        }
        faceDescriptors.push(new faceapi.LabeledFaceDescriptors(label, descriptors));
        Toastify({
            text: `Training xong data của ${label}!`
        }).showToast();
    }

    return faceDescriptors;
}

async function findExistingImagePath(basePath, extensions) {
    for (const ext of extensions) {
        const imagePath = `${basePath}.${ext}`;
        if (await fileExists(imagePath)) {
            return imagePath;
        }
    }
    return null;
}

async function fileExists(path) {
    try {
        await faceapi.fetchImage(path);
        return true;
    } catch (error) {
        return false;
    }
}

let faceMatcher;
async function init() {
    await Promise.all([
        faceapi.FaceLandmark68Net('/models'),
        faceapi.loadFaceLandmarkModel('/models'),
        faceapi.nets.ageGenderNet.load('/models'),
        faceapi.loadFaceExpressionModel('/models'),
        faceapi.loadSsdMobilenetv1Model('/models'),
        faceapi.loadFaceRecognitionModel('/models'),
        faceapi.nets.tinyFaceDetector.load('/models'),
        faceapi.nets.faceLandmark68Net.load("/models"),
    ]);

    const trainingData = await loadTrainingData();
    faceMatcher = new faceapi.FaceMatcher(trainingData, 0.5);

    document.querySelector("#loading").remove();
    Toastify({
        text: "Tải xong model nhận diện!",
    }).showToast();
}

init();

fileInput.addEventListener('change', handleFileInput);

async function handleFileInput() {
    const files = fileInput.files;

    const image = await faceapi.bufferToImage(files[0]);
    processImage(image);
}

function processImage(image) {
    const canvas = faceapi.createCanvasFromMedia(image);
    container.innerHTML = '';
    container.append(image);
    container.append(canvas);

    const size = {
        width: image.width,
        height: image.height
    };

    faceapi.matchDimensions(canvas, size);
    detectAndDraw(image, size, canvas);
}

async function detectAndDraw(source, size, drawCanvas) {
    const detections = await faceapi.detectAllFaces(source).withFaceLandmarks().withFaceDescriptors().withAgeAndGender();
    const resizedDetections = faceapi.resizeResults(detections, size);

    resizedDetections.forEach(result => {
        const { age, gender, genderProbability } = result
        new faceapi.draw.DrawTextField(
            [
                `${faceapi.utils.round(age, 0)} years`,
                `${gender} (${faceapi.utils.round(genderProbability)})`,
                `${faceMatcher.findBestMatch(result.descriptor).toString()}`
            ],
            result.detection.box.bottomLeft
        ).draw(drawCanvas)
    })

    for (const detection of resizedDetections) {
        const drawBox = new faceapi.draw.DrawBox(detection.detection.box, {
            // label: faceMatcher.findBestMatch(detection.descriptor).toString()
        })
        drawBox.draw(drawCanvas)
    }
}

function startVideo() {
    navigator.getUserMedia({ video: {} },
        stream => video.srcObject = stream,
        error => console.log(error)
    )
    container.innerHTML = '';
}

video.addEventListener('play', () => {
    const canvas = faceapi.createCanvasFromMedia(video)
    container.append(canvas)
    const size = {
        width: image.width,
        height: image.height
    };
    faceapi.matchDimensions(canvas, size);
    setInterval(async () => {
        const dectection = await faceapi.detectAllFaces(video, new faceapi
            .TinyFaceDetectorOptions()
            .withFaceLandmarks().
            widthFaceExpression().
            withFaceDescriptors())
        const resizedDetections = faceapi.resizeResults(dectection, size);
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
        faceapi.draw.drawDetections(canvas, resizedDetections);
        faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
        faceapi.draw.drawFaceExpressions(canvas, resizedDetections);
    }, 100)
})

function stopCamera() {
    if (video) {
        const tracks = video.getTracks();
        tracks.forEach(track => track.stop());
        container.innerHTML = '';
    }
}
