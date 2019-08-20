const f = 1000;
const imageWidth = 1080;
const imageHeight = 1920;
const c_x = imageWidth / 2;
const c_y = imageHeight / 2;

const objpoints = [
  [0.4, -1, 2],
  [-0.5, 0.5, 2],
  [1.5, -0.5, 4],
  [-0.4, -0.75, 1]
];
const imgpoints = [[740, 460], [290, 1210], [915, 835], [140, 210]];

const imgpoints2 = [[1290, 1210], [540, 2460], [1165, 1210], [1040, 3710]];

const imgpoints3 = [
  [185.8911133, 1000.3085938],
  [540, 1577.052002],
  [-145.110733, 1689.3399658],
  [487.3162842, 589.7321777]
];

const d1 = PnPSolver(f, f, c_x, c_y).solvePnP(objpoints, imgpoints3);

d1['rotation'] = [
  [d1['rotation'][0], d1['rotation'][3], d1['rotation'][6]],
  [d1['rotation'][1], d1['rotation'][4], d1['rotation'][7]],
  [d1['rotation'][2], d1['rotation'][5], d1['rotation'][8]]
];
var rot = d1['rotation'];
var rotMat3 = mat3.fromValues(
  rot[0][0],
  rot[1][0],
  rot[2][0],
  rot[0][1],
  rot[1][1],
  rot[2][1],
  rot[0][2],
  rot[1][2],
  rot[2][2]
);
var qt = quat.create();
quat.fromMat3(qt, rotMat3);

d1['translation'] = [
  [d1['translation'][0]],
  [d1['translation'][1]],
  [d1['translation'][2]]
];

console.log(qt);
console.log(d1);

// var data = d1;

// var newTranslateX = data["translation"][0][0];
// var newTranslateY = data["translation"][1][0];
// var newTranslateZ = data["translation"][2][0];

// var rot = data["rotation"];
