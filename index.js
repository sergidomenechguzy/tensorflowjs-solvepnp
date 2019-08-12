const tf = require("@tensorflow/tfjs");

const Ps = tf.tensor2d([
  [0.4, -1, 2, 1],
  [-0.5, 0.5, 2, 1],
  [1.5, -0.5, 4, 1],
  [-0.2, 0.4, 1, 1],
  [0.1, 1.1, 3, 1]
]);
// const ps = tf.tensor2d([
//   [2063.9523926, 438.5108948, 0.3202356],
//   [2527.7749023, 1448.7738037, -0.1905859],
//   [2461.2624512, -2307.2189941, -1.2053751],
//   [1571.5290527, 1802.5941162, -0.0869398],
//   [2612.279541, 1.8300147, -1.2407744]
// ]);
const ps = tf.tensor2d([
  [1480, 920, 2],
  [580, 2420, 2],
  [3660, 3340, 4],
	[340, 1360, 1],
	[1720, 3980, 3]
]);
// const ps = tf.tensor2d([[740, 460, 1], [290, 1210, 1], [915, 835, 1]]);

const f = 1000;
const imageWidth = 1080;
const imageHeight = 1920;
const c_x = imageWidth / 2;
const c_y = imageHeight / 2;
const A = tf.tensor2d([[f, 0, c_x], [0, f, c_y], [0, 0, 1]]);

// const r = tf.tensor1d([Math.random(), Math.random(), Math.random()]).variable();
// const t = tf
//   .tensor2d([[Math.random()], [Math.random()], [Math.random()]])
//   .variable();
const r = tf.tensor1d([2, 1, 3.5]).variable();
const t = tf.tensor2d([[0.6], [2], [0.3]]).variable();
// const r = tf.tensor1d([0, 0, 0]).variable();
// const t = tf.tensor2d([[0], [0], [0]]).variable();

const solvePNP = Ps => {
  const sin = tf.sin(r);
  const cos = tf.cos(r);
  const sin_pitch = sin.dot(tf.tensor1d([1, 0, 0]));
  const sin_yaw = sin.dot(tf.tensor1d([0, 1, 0]));
  const sin_roll = sin.dot(tf.tensor1d([0, 0, 1]));
  const cos_pitch = cos.dot(tf.tensor1d([1, 0, 0]));
  const cos_yaw = cos.dot(tf.tensor1d([0, 1, 0]));
  const cos_roll = cos.dot(tf.tensor1d([0, 0, 1]));

  const R_x = tf
    .tensor2d([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    .add(tf.mul(sin_pitch, tf.tensor2d([[0, 0, 0], [0, 0, -1], [0, 1, 0]])))
    .add(tf.mul(cos_pitch, tf.tensor2d([[0, 0, 0], [0, 1, 0], [0, 0, 1]])));
  const R_y = tf
    .tensor2d([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    .add(tf.mul(sin_yaw, tf.tensor2d([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])))
    .add(tf.mul(cos_yaw, tf.tensor2d([[1, 0, 0], [0, 0, 0], [0, 0, 1]])));
  const R_z = tf
    .tensor2d([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    .add(tf.mul(sin_roll, tf.tensor2d([[0, -1, 0], [1, 0, 0], [0, 0, 0]])))
    .add(tf.mul(cos_roll, tf.tensor2d([[1, 0, 0], [0, 1, 0], [0, 0, 0]])));

  // const R_x = tf.tensor2d([
  //   [1, 0, 0],
  //   [0, cos.get(0), -sin.get(0)],
  //   [0, sin.get(0), cos.get(0)]
  // ]);
  // const R_y = tf.tensor2d([
  //   [cos.get(1), 0, sin.get(1)],
  //   [0, 1, 0],
  //   [-sin.get(1), 0, cos.get(1)]
  // ]);
  // const R_z = tf.tensor2d([
  // 	[cos.get(2), -sin.get(2), 0],
  //   [sin.get(2), cos.get(2), 0],
  //   [0, 0, 1]
  // ]);

  const R = R_x.dot(R_y).dot(R_z);
  const Rt = tf.concat([R, t], 1);
  return A.dot(Rt)
    .dot(Ps.transpose())
    .transpose();
  // const data = res.bufferSync();
  // return res.div(data.get(2));
};

// const loss = (pred, label) => {
//   const res = label
//     .sub(pred)
//     .square()
//     .mean();
//   return res;
// };

// const trainNet = () => {
//   const learningRate = 0.1;
//   const optimizer = tf.train.adagrad(learningRate);

//   for (let i = 0; i < 5000; i++) {
//     const cost = optimizer.minimize(() => loss(solvePNP(Ps), ps), true);
//     // cost.print();
//   }

//   r.print();
//   t.print();
// };

// trainNet();
solvePNP(Ps).print();
