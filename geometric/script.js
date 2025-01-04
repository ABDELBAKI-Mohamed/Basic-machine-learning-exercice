// Canvas setup
const canvas = document.getElementById('plot');
const ctx = canvas.getContext('2d');

// Training data: Action and Comedy scores
const data = [
  { actionScore: 8, comedyScore: 3, label: 1, name: "Action Movie A" },
  { actionScore: 6, comedyScore: 7, label: 1, name: "Comedy Movie B" },
  { actionScore: 7, comedyScore: 5, label: 1, name: "Action-Comedy C" },
  { actionScore: 2, comedyScore: 8, label: 0, name: "Romantic Comedy D" },
  { actionScore: 3, comedyScore: 6, label: 0, name: "Light Comedy E" },
  { actionScore: 1, comedyScore: 9, label: 0, name: "Drama F" }
];

// Logistic regression parameters
let weights = {
  w1: Math.random() * 0.1,
  w2: Math.random() * 0.1,
  bias: Math.random() * 0.1
};
let epoch = 0;
const learningRate = 0.1;

// Normalize data between 0 and 1
function normalize(value, min, max) {
  return (value - min) / (max - min);
}

// Convert data coordinates to canvas coordinates
function toCanvasCoords(x, y) {
  const padding = 40;
  const width = canvas.width - 2 * padding;
  const height = canvas.height - 2 * padding;

  return {
    x: padding + x * width,
    y: canvas.height - (padding + y * height)
  };
}

// Step function
function step(z) {
  return z >= 0 ? 1 : 0;
}

// Predict using the step function
function predict(actionScore, comedyScore) {
  const x1 = normalize(actionScore, 1, 10);
  const x2 = normalize(comedyScore, 1, 10);
  const z = weights.w1 * x1 + weights.w2 * x2 + weights.bias;
  return step(z); // Step function replaces sigmoid
}

// Train one step
function trainStep() {
  let updated = false;

  data.forEach(point => {
    const x1 = normalize(point.actionScore, 1, 10);
    const x2 = normalize(point.comedyScore, 1, 10);
    const prediction = predict(point.actionScore, point.comedyScore);
    const error = point.label - prediction;

    // Gradient descent updates
    weights.w1 += learningRate * error * x1;
    weights.w2 += learningRate * error * x2;
    weights.bias += learningRate * error;

    updated = true;
  });

  if (updated) {
    epoch++;
    document.getElementById('epoch').textContent = `Epoch: ${epoch}`;
    document.getElementById('weights').textContent =
      `Weights: w1=${weights.w1.toFixed(3)}, w2=${weights.w2.toFixed(3)}, bias=${weights.bias.toFixed(3)}`;
    draw();
  }
}

// Reset the model
function reset() {
  // Randomize weights and reset epoch
  weights = {
    w1: Math.random() - 0.5, // Range: -0.5 to 0.5
    w2: Math.random() - 0.5,
    bias: Math.random() - 0.5
  };
  epoch = 0;

  // Update UI
  document.getElementById('epoch').textContent = `Epoch: ${epoch}`;
  document.getElementById('weights').textContent =
    `Weights: w1=${weights.w1.toFixed(3)}, w2=${weights.w2.toFixed(3)}, bias=${weights.bias.toFixed(3)}`;
  
  draw();
}

// Draw everything
function draw() {
  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw axes
  ctx.strokeStyle = '#ccc';
  ctx.beginPath();
  ctx.moveTo(40, canvas.height - 40); // X-axis
  ctx.lineTo(canvas.width - 40, canvas.height - 40);
  ctx.moveTo(40, 40); // Y-axis
  ctx.lineTo(40, canvas.height - 40);
  ctx.stroke();

  // Draw points
  data.forEach(point => {
    const x = normalize(point.actionScore, 1, 10);
    const y = normalize(point.comedyScore, 1, 10);
    const coords = toCanvasCoords(x, y);

    ctx.beginPath();
    ctx.arc(coords.x, coords.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = point.label === 1 ? '#4444ff' : '#ff8800';
    ctx.fill();
  });

  // Draw decision boundary
  drawDecisionBoundary();

  // Draw axis labels
  ctx.fillStyle = '#000';
  ctx.font = '12px Arial';
  ctx.fillText('Action Score', canvas.width / 2, canvas.height - 10);
  ctx.save();
  ctx.translate(15, canvas.height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Comedy Score', 0, 0);
  ctx.restore();
}

// Draw decision boundary
function drawDecisionBoundary() {
  ctx.strokeStyle = '#4CAF50';
  ctx.beginPath();

  for (let x = 0; x <= 1; x += 0.01) {
    const y = -(weights.w1 * x + weights.bias) / weights.w2;
    const coords = toCanvasCoords(x, y);

    if (x === 0) {
      ctx.moveTo(coords.x, coords.y);
    } else {
      ctx.lineTo(coords.x, coords.y);
    }
  }
  ctx.stroke();
}

// Initial draw
draw();
