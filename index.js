const tf = require('@tensorflow/tfjs');

const f = 1000;
const imageWidth = 1080;
const imageHeight = 1920;
const c_x = imageWidth / 2;
const c_y = imageHeight / 2;
const A = tf.tensor2d([[f, 0, c_x], [0, f, c_y], [0, 0, 1]]);
const Ps = tf.tensor2d([
  [0.4, -1, 2, 1],
  [-0.5, 0.5, 2, 1],
  [1.5, -0.5, 4, 1],
  [-0.4, -0.75, 1, 1]
]);

// rotation 0 and translation 0
// const r = tf.tensor1d([0, 0, 0]);
// const r_x = tf.scalar(0);
// const r_y = tf.scalar(0);
// const r_z = tf.scalar(0);
// const t = tf.tensor2d([[0], [0], [0]]);
// const ps = tf.tensor2d([
//   [740, 460, 1],
//   [290, 1210, 1],
//   [915, 835, 1],
//   [140, 210, 1]
// ]);

// rotation 0 and translation set
// const r = tf.tensor1d([0, 0, 0]);
// const r_x = tf.scalar(0);
// const r_y = tf.scalar(0);
// const r_z = tf.scalar(0);
// const t = tf.tensor2d([[0.5], [1.3], [-0.8]]);
const ps = tf.tensor2d([
  [1290, 1210, 1],
  [540, 2460, 1],
  [1165, 1210, 1],
  [1040, 3710, 1]
]);

// one rotation set and translation set
// const r = tf.tensor1d([2, 0, 0]);
// const r_x = tf.scalar(2);
// const r_y = tf.scalar(0);
// const r_z = tf.scalar(0);
// const t = tf.tensor2d([[0.5], [1.3], [-0.8]]);
// const ps = tf.tensor2d([
//   [185.8911133, 1000.3085938, 1],
//   [540, 1577.052002, 1],
//   [-145.110733, 1689.3399658, 1],
//   [487.3162842, 589.7321777, 1]
// ]);

// random rotation and translation for training
// const r = tf.tensor1d([Math.random(), Math.random(), Math.random()]).variable();
const r_x = tf.scalar(Math.random()).variable();
const r_y = tf.scalar(Math.random()).variable();
const r_z = tf.scalar(Math.random()).variable();
const t = tf
  .tensor2d([[Math.random()], [Math.random()], [Math.random()]])
  .variable();

// OpenCV explanation: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#rodrigues
// OpenCV implementation: https://github.com/opencv/opencv/blob/master/modules/calib3d/test/test_fundam.cpp#L48
// For value comparison: https://www.andre-gaschler.com/rotationconverter/
const rodrigues = (r_x, r_y, r_z) => {
  const r = tf.stack([r_x, r_y, r_z]);
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
  const w_x = itheta.mul(r_x);
  const w_y = itheta.mul(r_y);
  const w_z = itheta.mul(r_z);
  const w = tf.stack([w_x, w_y, w_z]);
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

// const solvePNP = Ps => {
//   const sin = tf.sin(r);
//   const cos = tf.cos(r);
//   const sin_pitch = sin.dot(tf.tensor1d([1, 0, 0]));
//   const sin_yaw = sin.dot(tf.tensor1d([0, 1, 0]));
//   const sin_roll = sin.dot(tf.tensor1d([0, 0, 1]));
//   const cos_pitch = cos.dot(tf.tensor1d([1, 0, 0]));
//   const cos_yaw = cos.dot(tf.tensor1d([0, 1, 0]));
//   const cos_roll = cos.dot(tf.tensor1d([0, 0, 1]));

// const R_x = tf
//   .tensor2d([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
//   .add(tf.mul(sin_pitch, tf.tensor2d([[0, 0, 0], [0, 0, -1], [0, 1, 0]])))
//   .add(tf.mul(cos_pitch, tf.tensor2d([[0, 0, 0], [0, 1, 0], [0, 0, 1]])));
//   const R_y = tf
//     .tensor2d([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
//     .add(tf.mul(sin_yaw, tf.tensor2d([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])))
//     .add(tf.mul(cos_yaw, tf.tensor2d([[1, 0, 0], [0, 0, 0], [0, 0, 1]])));
//   const R_z = tf
//     .tensor2d([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
//     .add(tf.mul(sin_roll, tf.tensor2d([[0, -1, 0], [1, 0, 0], [0, 0, 0]])))
//     .add(tf.mul(cos_roll, tf.tensor2d([[1, 0, 0], [0, 1, 0], [0, 0, 0]])));

//   const R = R_x.dot(R_y).dot(R_z);
//   const Rt = tf.concat([R, t], 1);
//   return A.dot(Rt)
//     .dot(Ps.transpose())
//     .transpose();
// };

// implementation with no rotation and only translation
const solvePNPNoR = P => {
  const R = tf.tensor2d([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  const Rt = tf.concat([R, t], 1);
  const res = A.dot(Rt)
    .dot(P.transpose())
    .transpose();
  const w = tf.tensor1d([0, 0, 1]).dot(res);
  return res.div(w);
};

const solvePNPNoRMultiple = Ps => {
  const [p0, p1, p2, p3] = tf.unstack(Ps);
  return tf.stack([
    solvePNPNoR(p0),
    solvePNPNoR(p1),
    solvePNPNoR(p2),
    solvePNPNoR(p3)
  ]);
};

const trainNetNoR = () => {
  const learningRate = 0.05;
  const optimizer = tf.train.adamax(learningRate);

  for (let i = 0; i < 500; i++) {
    const cost = optimizer.minimize(
      () => tf.losses.meanSquaredError(ps, solvePNPNoRMultiple(Ps)),
      true
    );
    cost.print();
  }

  console.log('=========================================');
  console.log('Translation:');
  t.print();
};

// implementation with rodrigues rotation and translation
const solvePNPWithR = P => {
  // const R = rodrigues(r_x, tf.scalar(0), tf.scalar(0));
  const R = rodrigues(r_x, r_y, r_z);
  R.print();
  const Rt = tf.concat([R, t], 1);
  const res = A.dot(Rt)
    .dot(P.transpose())
    .transpose();
  const w = tf.tensor1d([0, 0, 1]).dot(res);
  return res.div(w);
};

const solvePNPWithRMultiple = Ps => {
  const [p0, p1, p2, p3] = tf.unstack(Ps);
  return tf.stack([
    solvePNPWithR(p0),
    solvePNPWithR(p1),
    solvePNPWithR(p2),
    solvePNPWithR(p3)
  ]);
};

const trainNetWithR = () => {
  const learningRate = 0.1;
  const optimizer = tf.train.adamax(learningRate);

  r_x.print();
  t.print();

  for (let i = 0; i < 3000; i++) {
    const cost = optimizer.minimize(
      () => tf.losses.meanSquaredError(ps, solvePNPWithRMultiple(Ps)),
      true
    );
    cost.print();
  }

  console.log('=========================================');
  console.log('Rotation X:');
  r_x.print();
  console.log('Rotation Y:');
  r_y.print();
  console.log('Rotation Z:');
  r_z.print();
  console.log('Translation:');
  t.print();
};

trainNetNoR();
// trainNetWithR();
console.log('Prediction:');
solvePNPNoRMultiple(Ps).print();
// solvePNPWithRMultiple(Ps).print();
console.log('Label:');
ps.print();
