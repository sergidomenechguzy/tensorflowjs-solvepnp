const createRotationMatrix = (pitch, yaw, roll) => {
  const r = tf.tensor1d([pitch, yaw, roll]);
  const sin = tf.sin(r).bufferSync();
  const cos = tf.cos(r).bufferSync();

  const R_x = tf.tensor2d([
    [1, 0, 0],
    [0, cos.get(0), -sin.get(0)],
    [0, sin.get(0), cos.get(0)]
  ]);
  const R_y = tf.tensor2d([
    [cos.get(1), 0, sin.get(1)],
    [0, 1, 0],
    [-sin.get(1), 0, cos.get(1)]
  ]);
  const R_z = tf.tensor2d([
    [cos.get(2), -sin.get(2), 0],
    [sin.get(2), cos.get(2), 0],
    [0, 0, 1]
  ]);

  return R_x.dot(R_y).dot(R_z);
};

const convertTo2d = async (point3d, f, c_x, c_y) => {
  const pointData = await point3d.buffer();
  const x = f * (pointData.get(0) / pointData.get(2)) + c_x;
  const y = f * (pointData.get(1) / pointData.get(2)) + c_y;
  return tf.tensor1d([x, y, 1]);
};

const get2dPoints = async () => {
  const p1 = await convertTo2d(P1, f, c_x, c_y);
  const p2 = await convertTo2d(P2, f, c_x, c_y);
  const p3 = await convertTo2d(P3, f, c_x, c_y);

  p1.print(); // [740, 460, 1]
  p2.print(); // [290, 1210, 1]
  p3.print(); // [915, 835, 1]
};

const Ps = tf.tensor2d([
  [0.4, -1, 2, 1],
  [-0.5, 0.5, 2, 1],
  [1.5, -0.5, 4, 1]
  [-0.2, 0.4, 1, 1],
  [0.1, 1.1, 3, 1]
]);