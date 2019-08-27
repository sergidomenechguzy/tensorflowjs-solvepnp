// const tf = require('@tensorflow/tfjs');
// const tf = require('@tensorflow/tfjs-node');
// require('@tensorflow/tfjs-node-gpu');

const plotData = {
  r_x: [],
  r_y: [],
  r_z: [],
  t_x: [],
  t_y: [],
  t_z: [],
  loss: []
};

const f = 1000;
const imageWidth = 1080;
const imageHeight = 1920;
const c_x = imageWidth / 2;
const c_y = imageHeight / 2;
const A = tf.tensor2d([[f, 0, c_x], [0, f, c_y], [0, 0, 1]]);

// const Ps = tf.tensor2d([
//   [0.4, -1, 2],
//   [-0.5, 0.5, 2],
//   [1.5, -0.5, 4],
//   [-0.4, -0.75, 1]
// ]);
// const Ps = tf.tensor2d([[1, 0, 3], [0, 1, 3], [0, 0, 4], [0, 0, 3], [1, 1, 4]]);

const Ps = tf.randomUniform([10, 3], 0.1, 5);

Ps.print();

// start rotation and translation values for training
const r_train = tf.tensor1d([1, 0.3, -0.2]).variable();
const t_train = tf.tensor1d([0.8, 0.3, 2]).variable();

// rotation and translation values for calculation
const r = tf.tensor1d([0, 0.1, 0]);
const t = tf.tensor1d([-0.3, 0, 3]);

// OpenCV explanation: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#rodrigues
// OpenCV implementation: https://github.com/opencv/opencv/blob/master/modules/calib3d/test/test_fundam.cpp#L48
// For value comparison: https://www.andre-gaschler.com/rotationconverter/
const rodrigues = r => {
  const theta = tf.norm(r);
  const itheta = tf.scalar(1).div(theta);
  if (
    itheta
      .isInf()
      .bufferSync()
      .get(0)
  ) {
    return tf.eye(3, 3);
  }
  const w = itheta.mul(r);
  const [w_x, w_y, w_z] = w.unstack();
  const alpha = theta.cos();
  const beta = theta.sin();
  const gamma = tf.scalar(1).sub(alpha);
  const omegav = w_x
    .mul(tf.tensor2d([[0, 0, 0], [0, 0, -1], [0, 1, 0]]))
    .add(w_y.mul(tf.tensor2d([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])))
    .add(w_z.mul(tf.tensor2d([[0, -1, 0], [1, 0, 0], [0, 0, 0]])));
  const A = tf.outerProduct(w, w);
  return tf
    .eye(3, 3)
    .mul(alpha)
    .add(A.mul(gamma))
    .add(omegav.mul(beta));
};

// Calculation of 2d points using rodrigues rotation and translation vector
const calculate2dPoints = (Ps, r = r_train, t = t_train) => {
  const R = rodrigues(r);
  const Rt = tf.concat([R, t.expandDims(1)], 1);
  const res = A.dot(Rt).dot(Ps.transpose().pad([[0, 1], [0, 0]], 1));

  return res
    .div(res.slice([2], [1]))
    .slice([0], [2])
    .transpose();
};

const solvePnP = ps => {
  const start = Date.now();
  let learningRate = 0.0003;
  let rho = 0.8;
  let optimizer = tf.train.rmsprop(learningRate, rho, 0.0, null, true);
  let loss;
  let i = 0;

  for (i; i < 4000; i++) {
    const r_buffer = r_train.bufferSync();
    const t_buffer = t_train.bufferSync();
    plotData.r_x.push({ x: i, y: r_buffer.get(0) });
    plotData.r_y.push({ x: i, y: r_buffer.get(1) });
    plotData.r_z.push({ x: i, y: r_buffer.get(2) });
    plotData.t_x.push({ x: i, y: t_buffer.get(0) });
    plotData.t_y.push({ x: i, y: t_buffer.get(1) });
    plotData.t_z.push({ x: i, y: t_buffer.get(2) });

    loss = optimizer.minimize(
      () => tf.losses.meanSquaredError(ps, calculate2dPoints(Ps)),
      true
    );
    plotData.loss.push({ x: i, y: loss.bufferSync().get(0) });

    if (i % 100 === 0) {
      console.log('LOSS:');
      loss.print();
    }
    if (
      loss
        .less(tf.scalar(0.5))
        .bufferSync()
        .get(0)
    ) {
      break;
    }
  }
  const ps_pred = calculate2dPoints(Ps);
  const diff = Date.now() - start;
  return [ps_pred, loss, i, diff];
};

r_start = r_train.clone();
t_start = t_train.clone();

const ps_label = calculate2dPoints(Ps, r, t);
const [ps_pred, loss, iterations, duration] = solvePnP(ps_label);

console.log('\n===========================================\n');
console.log('Start Rotation:');
r_start.print();
console.log('Final Rotation:');
r_train.print();
console.log('Actual Rotation:');
r.print();
console.log('\n===========================================\n');
console.log('Start Translation:');
t_start.print();
console.log('Final Translation:');
t_train.print();
console.log('Actual Translation:');
t.print();
console.log('\n===========================================\n');
console.log('Prediction:');
ps_pred.print();
console.log('Label:');
ps_label.print();
console.log('\n===========================================\n');
console.log(`Iterations: ${iterations}`);
console.log(`Duration: ${duration}ms`);
console.log('Distance:');
loss.print();

const renderPlot = () => {
  const rotationSurface = tfvis
    .visor()
    .surface({ name: 'Rotation', tab: 'Values' });
  tfvis.render.linechart(
    rotationSurface,
    { values: [plotData.r_x, plotData.r_y, plotData.r_z] },
    {}
  );
  const translationSurface = tfvis
    .visor()
    .surface({ name: 'Translation', tab: 'Values' });
  tfvis.render.linechart(
    translationSurface,
    { values: [plotData.t_x, plotData.t_y, plotData.t_z] },
    {}
  );
  const lossSurface = tfvis.visor().surface({ name: 'Loss', tab: 'Values' });
  tfvis.render.linechart(lossSurface, { values: [plotData.loss] }, {});
};

document.addEventListener('DOMContentLoaded', renderPlot);
