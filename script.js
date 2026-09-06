        // ─── GLOBALS ───────────────────────────────────────────────
        const MODEL_KEY = 'digit_model_v1';
        const INPUT_SIZE = 784;
        const NUM_CLASSES = 10;
        const EPOCHS = 10;
        const BATCH_SIZE = 256;

        let model = null;
        let training = false;


        // Canvas drawing
        const canvas = document.getElementById('drawCanvas');
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, 280, 280);
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 24;

        let drawing = false;
        let lastX = 0, lastY = 0;

        canvas.addEventListener('mousedown', e => {
            drawing = true;
            const rect = canvas.getBoundingClientRect();
            lastX = e.clientX - rect.left;
            lastY = e.clientY - rect.top;
        });
        canvas.addEventListener('mousemove', e => {
            if (!drawing) return;
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(x, y);
            ctx.stroke();
            lastX = x; lastY = y;
        });
        canvas.addEventListener('mouseup', () => drawing = false);
        canvas.addEventListener('mouseleave', () => drawing = false);

        // Touch support
        canvas.addEventListener('touchstart', e => {
            e.preventDefault();
            drawing = true;
            const rect = canvas.getBoundingClientRect();
            lastX = e.touches[0].clientX - rect.left;
            lastY = e.touches[0].clientY - rect.top;
        });
        canvas.addEventListener('touchmove', e => {
            e.preventDefault();
            if (!drawing) return;
            const rect = canvas.getBoundingClientRect();
            const x = e.touches[0].clientX - rect.left;
            const y = e.touches[0].clientY - rect.top;
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(x, y);
            ctx.stroke();
            lastX = x; lastY = y;
        });
        canvas.addEventListener('touchend', () => drawing = false);

        // ─── MODEL BUILD ───────────────────────────────────────────
        function buildModel() {
            const m = tf.sequential();
            m.add(tf.layers.conv2d({ inputShape: [28, 28, 1], filters: 32, kernelSize: 3, activation: 'relu' }));
            m.add(tf.layers.maxPooling2d({ poolSize: 2 }));
            m.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }));
            m.add(tf.layers.maxPooling2d({ poolSize: 2 }));
            m.add(tf.layers.flatten());
            m.add(tf.layers.dense({ units: 128, activation: 'relu' }));
            m.add(tf.layers.dropout({ rate: 0.3 }));
            m.add(tf.layers.dense({ units: NUM_CLASSES, activation: 'softmax' }));
            m.compile({
                optimizer: tf.train.adam(0.001),
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy']
            });
            return m;
        }

        // ─── MNIST DATA LOADING ────────────────────────────────────
        async function loadMNIST() {
            const url = 'https://storage.googleapis.com/tfjs-tutorials/mnist';
            const [trainData, testData] = await Promise.all([
                fetch(`${url}/train-images-uint8.npy`).then(r => r.arrayBuffer()),
                fetch(`${url}/train-labels-uint8.npy`).then(r => r.arrayBuffer()),
                fetch(`${url}/test-images-uint8.npy`).then(r => r.arrayBuffer()),
                fetch(`${url}/test-labels-uint8.npy`).then(r => r.arrayBuffer())
            ]);
            // Parse .npy format (10-byte header)
            function parseNpy(buffer) {
                const view = new DataView(buffer);
                let headerLen = 10; // skip magic
                // find end of header dict
                let i = 10;
                while (view.getUint8(i) !== 0x7d) i++; // '}'
                i++;
                return new Uint8Array(buffer, i);
            }
            const trainImages = parseNpy(trainData);
            const trainLabels = parseNpy(trainData.slice ? trainData : new ArrayBuffer(0));
            // Actually let's use a simpler approach with the tfjs MNIST helper
            return { trainImages, trainLabels, testData };
        }

        // Simpler: use the well-known MNIST helper
        async function fetchMNISTData() {
            const IMAGE_SIZE = 784;
            const NUM_CLASSES = 10;
            const NUM_DATASET_ELEMENTS = 65000;
            const NUM_TRAIN_ELEMENTS = 55000;
            const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS; // 10000

            const IMAGES_URL = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
            const LABELS_URL = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

            const [imgResponse, labelsResponse] = await Promise.all([
                fetch(IMAGES_URL),
                fetch(LABELS_URL)
            ]);

            const labels = new Uint8Array(await labelsResponse.arrayBuffer());
            // labels.length = 650000 (65000 × 10 one-hot bytes)


            const imgBlob = await imgResponse.blob();
            const img = await createImageBitmap(imgBlob);
            console.log('Sprite dimensions:', img.width, 'x', img.height);

            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d', { willReadFrequently: true });
            ctx.drawImage(img, 0, 0);
            const allPixels = ctx.getImageData(0, 0, img.width, img.height).data;

            const images = new Float32Array(NUM_DATASET_ELEMENTS * 784);
            for (let i = 0; i < NUM_DATASET_ELEMENTS; i++) {
                for (let j = 0; j < 784; j++) {
                    images[i * 784 + j] = allPixels[(i * 784 + j) * 4] / 255;
                }
            }

            // Slice images (flat 784 per image)
            const trainImages = images.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
            const testImages = images.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);

            // Slice labels (one-hot, 10 bytes per image)
            const trainLabels = labels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
            const testLabels = labels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);

            // Tensors — labels are ALREADY one-hot, no tf.oneHot needed
            const xs = tf.tensor4d(trainImages, [NUM_TRAIN_ELEMENTS, 28, 28, 1]);
            const testXs = tf.tensor4d(testImages, [NUM_TEST_ELEMENTS, 28, 28, 1]);

            const ys = tf.tensor2d(trainLabels, [NUM_TRAIN_ELEMENTS, NUM_CLASSES]);
            const testYs = tf.tensor2d(testLabels, [NUM_TEST_ELEMENTS, NUM_CLASSES]);
            // DIAGNOSTIC — check if images are actually loaded
            const sample = xs.dataSync();
            const nonZero = sample.filter(v => v > 0.01).length;
            return { xs, ys, testXs, testYs };
        }

        // ─── SAVE / LOAD MODEL (localStorage) ─────────────────────
        async function saveModel() {
            await model.save('indexeddb://digit_model');
            log('✓ Model saved to IndexedDB');
            updateStatus('Model saved ✓');
        }

        async function loadModel() {
            try {
                model = await tf.loadLayersModel('indexeddb://digit_model');
                document.getElementById('modelStatus').textContent = 'Model loaded ✓';
                document.getElementById('trainStatus').textContent = 'Model loaded from storage';

                return true;
            } catch (e) {
                console.error('Failed to load model:', e);
                return false;
            }
        }

        async function deleteModel() {
            await tf.io.removeIOHandler('indexeddb://digit_model');
            model = null;

            document.getElementById('modelStatus').textContent = 'No model loaded';
            document.getElementById('trainStatus').textContent = 'Model deleted';
            document.getElementById('result').textContent = '';
        }

        // ─── TRAINING ──────────────────────────────────────────────
        async function trainModel() {
            if (training) return;
            training = true;
            document.getElementById('trainBtn').disabled = true;
            document.getElementById('trainingLog').textContent = 'Loading MNIST data...\n';
            document.getElementById('trainStatus').textContent = 'Loading data...';

            try {
                const { xs, ys, testXs, testYs } = await fetchMNISTData();
                document.getElementById('trainStatus').textContent = 'Training...';

                model = buildModel();

                let lastValAcc = 0;
                let bestValAcc = 0;
                let bestWeights = null;
                await model.fit(xs, ys, {
                    epochs: EPOCHS,
                    batchSize: BATCH_SIZE,
                    validationData: [testXs, testYs],
                    shuffle: true,
                    callbacks: {
                        onEpochEnd: async (epoch, logs) => {
                            const acc = (logs.acc * 100).toFixed(2);
                            const loss = logs.loss.toFixed(4);
                            const valAcc = (logs.val_acc * 100).toFixed(2);
                            const valLoss = logs.val_loss.toFixed(4);
                            log(`Epoch ${epoch + 1}/${EPOCHS} | Loss: ${loss} | Acc: ${acc}% | Val Loss: ${valLoss} | Val Acc: ${valAcc}%`);

                            lastValAcc = logs.val_acc;

                            lossData.push(logs.loss);
                            accData.push(logs.acc);
                            valLossData.push(logs.val_loss);
                            valAccData.push(logs.val_acc);
                            drawChart('lossCanvas', lossData, valLossData, 'Loss', '#e94560', '#ff9f43');
                            drawChart('accCanvas', accData, valAccData, 'Accuracy', '#00ff88', '#48dbfb');
                        }
                    }
                });

                if (bestWeights) {
                    model.setWeights(bestWeights);
                    bestWeights.forEach(w => w.dispose());
                }

                xs.dispose(); ys.dispose();
                testXs.dispose(); testYs.dispose();

                await saveModel();
                document.getElementById('trainStatus').textContent = `Training complete! (Val Acc: ${(lastValAcc * 100).toFixed(2)}%)`;  // ← USE lastValAcc, NOT history
                log('\n✓ Training complete. Model saved.');

            } catch (e) {
                log('\n✗ Error: ' + e.message);
                document.getElementById('trainStatus').textContent = 'Error: ' + e.message;
            }
            training = false;
            document.getElementById('trainBtn').disabled = false;
        }

        // ─── PREDICTION ────────────────────────────────────────────
        async function predict() {
            try {
                if (!model) {
                    document.getElementById('result').textContent = 'No model loaded';
                    document.getElementById('result').style.color = '#e94560';
                    return;
                }

                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = 28; tempCanvas.height = 28;
                const tctx = tempCanvas.getContext('2d');
                tctx.fillStyle = 'black';
                tctx.fillRect(0, 0, 28, 28);
                tctx.drawImage(canvas, 0, 0, 280, 280, 0, 0, 28, 28);

                const imgData = tctx.getImageData(0, 0, 28, 28);
                const pixels = new Float32Array(784);
                for (let i = 0; i < 784; i++) {
                    pixels[i] = imgData.data[i * 4] / 255;
                }

                const input = tf.tensor4d(pixels, [1, 28, 28, 1]);
                const prediction = model.predict(input);
                const scores = await prediction.data();
                prediction.dispose();
                input.dispose();

                const maxIdx = Array.from(scores).indexOf(Math.max(...scores));

                document.getElementById('result').textContent = maxIdx;
                document.getElementById('result').style.color = '#00ff88';

                const confBar = document.getElementById('confBar');
                confBar.innerHTML = '';
                const colors = ['#e94560', '#ff9f43', '#feca57', '#48dbfb', '#0abde3', '#10ac84', '#00d4ff', '#5352ed', '#a29bfe', '#fd79a8'];
                scores.forEach((s, i) => {
                    const div = document.createElement('div');
                    div.style.width = (s * 100) + '%';
                    div.style.background = colors[i];
                    confBar.appendChild(div);
                });


            } catch (e) {
                document.getElementById('result').textContent = 'Error: ' + e.message;
                document.getElementById('result').style.color = '#e94560';
                console.error('Predict error:', e);
            }
        }

        function clearCanvas() {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, 280, 280);
            document.getElementById('result').textContent = '';
            document.getElementById('confBar').innerHTML = '';
        }

        // ─── TRAINING LOG ──────────────────────────────────────────
        function log(msg) {
            const el = document.getElementById('trainingLog');
            el.textContent += msg + '\n';
            el.scrollTop = el.scrollHeight;
        }

        function updateStatus(msg) {
            document.getElementById('modelStatus').textContent = msg;
        }

        // ─── GRAPH (tfjs-vis) ──────────────────────────────────────
        let lossData = [];
        let accData = [];
        let valLossData = [];
        let valAccData = [];

        function updateGraph(epoch, logs) {
            lossData.push(logs.loss);
            accData.push(logs.acc);
            valLossData.push(logs.val_loss);
            valAccData.push(logs.val_acc);

            drawChart('lossCanvas', lossData, valLossData, 'Loss', '#e94560', '#ff9f43');
            drawChart('accCanvas', accData, valAccData, 'Accuracy', '#00ff88', '#48dbfb');
        }
        function drawChart(canvasId, trainData, valData, title, c1, c2) {
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;

            const scale = 2; // retina factor
            const W = canvas.width, H = canvas.height;
            const ctx = canvas.getContext('2d');
            ctx.scale(scale, scale);

            const w = W / scale, h = H / scale;
            const pad = { top: 25, right: 10, bottom: 25, left: 45 };
            const plotW = w - pad.left - pad.right;
            const plotH = h - pad.top - pad.bottom;

            ctx.clearRect(0, 0, w, h);
            ctx.fillStyle = '#16213e';
            ctx.fillRect(0, 0, w, h);

            ctx.fillStyle = '#fff';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(title, w / 2, 15);

            if (trainData.length < 2) return;

            const allVals = [...trainData, ...valData];
            const maxVal = Math.max(...allVals) * 1.1;
            const minVal = 0;
            const numPoints = trainData.length;

            ctx.strokeStyle = '#555';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(pad.left, pad.top);
            ctx.lineTo(pad.left, h - pad.bottom);
            ctx.lineTo(w - pad.right, h - pad.bottom);
            ctx.stroke();

            ctx.fillStyle = '#aaa';
            ctx.font = '9px monospace';
            ctx.textAlign = 'right';
            for (let i = 0; i <= 4; i++) {
                const val = minVal + (maxVal - minVal) * (i / 4);
                const y = h - pad.bottom - (plotH * i / 4);
                ctx.fillText(val.toFixed(2), pad.left - 5, y + 3);
                ctx.strokeStyle = '#333';
                ctx.beginPath();
                ctx.moveTo(pad.left, y);
                ctx.lineTo(w - pad.right, y);
                ctx.stroke();
            }

            ctx.textAlign = 'center';
            for (let i = 0; i < numPoints; i += Math.ceil(numPoints / 5)) {
                const x = pad.left + (plotW * i / (numPoints - 1));
                ctx.fillText(i + 1, x, h - pad.bottom + 14);
            }

            function drawLine(data, color) {
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.beginPath();
                for (let i = 0; i < data.length; i++) {
                    const x = pad.left + (plotW * i / (numPoints - 1));
                    const y = h - pad.bottom - (plotH * (data[i] - minVal) / (maxVal - minVal));
                    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                }
                ctx.stroke();
            }

            drawLine(trainData, c1);
            drawLine(valData, c2);

            ctx.font = '10px sans-serif';
            ctx.textAlign = 'left';
            ctx.fillStyle = c1;
            ctx.fillText('● Train', w - pad.right - 80, 15);
            ctx.fillStyle = c2;
            ctx.fillText('● Val', w - pad.right - 35, 15);

            ctx.setTransform(1, 0, 0, 1, 0, 0); // reset scale for next call
        }
        async function debugData() {
            const { xs, ys } = await fetchMNISTData();

            // Get first 5 images and their labels
            const imgData = await xs.slice([0, 0, 0], [5, 28, 28]).data();
            const labelData = await ys.slice([0, 0], [5, 10]).data();

            for (let i = 0; i < 5; i++) {
                const label = labelData[i * 10].toFixed(0); // one-hot → index
                const img = imgData.slice(i * 784, (i + 1) * 784);
                const nonZero = img.filter(v => v > 0.01).length;
                console.log(`Image ${i}: label=${label}, non-zero pixels=${nonZero}`);

                // Draw first image on the canvas for visual check
                if (i === 0) {
                    const tempCanvas = document.createElement('canvas');
                    tempCanvas.width = 28; tempCanvas.height = 28;
                    const tctx = tempCanvas.getContext('2d');
                    const imageData = tctx.createImageData(28, 28);
                    for (let p = 0; p < 784; p++) {
                        const val = Math.floor(img[p] * 255);
                        imageData.data[p * 4] = val;
                        imageData.data[p * 4 + 1] = val;
                        imageData.data[p * 4 + 2] = val;
                        imageData.data[p * 4 + 3] = 255;
                    }
                    tctx.putImageData(imageData, 0, 0);
                    // Draw it big on the main canvas
                    ctx.fillStyle = 'black';
                    ctx.fillRect(0, 0, 280, 280);
                    ctx.imageSmoothingEnabled = false;
                    ctx.drawImage(tempCanvas, 0, 0, 28, 28, 0, 0, 280, 280);
                }
            }

            xs.dispose(); ys.dispose();
        }

        // ─── INIT ──────────────────────────────────────────────────
        (async function init() {
            // Check if model exists
            const hasModel = await loadModel();
            if (hasModel) {
                document.getElementById('trainStatus').textContent = 'Existing model found – no retraining needed';
            } else {
                document.getElementById('trainStatus').textContent = 'No model found – click Train to start';
            }
        })();
