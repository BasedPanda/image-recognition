const tf = require('@tensorflow/tfjs-node');
const cocoSsd = require('@tensorflow-models/coco-ssd');
const express = require('express');
const multer = require('multer');

const app = express();
const upload = multer({ dest: 'uploads/' });

app.use(express.static('public'));

app.post('/upload', upload.single('image'), async (req, res) => {
  const image = tf.node.decodeImage(req.file.buffer);
  const model = await cocoSsd.load();

  const predictions = await model.detect(image);
  console.log(predictions);

  res.send(predictions);
});

app.listen(3000, () => {
  console.log('listening on *:3000');
});
