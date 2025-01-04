 // =============== Paramètres ===============
 let currentEpoch = 1;
 const maxEpochs = 200;    // e.g. 200
 let currentSampleIndex = 0;
 const learningRate = 0.5; // e.g. 0.5

 // Jeu de données : [Action, Comedy, Duration], target
 const trainingData = [
   { input: [7, 3, 2], target: 0.8 },
   { input: [3, 9, 1], target: 0.6 },
   { input: [8, 2, 3], target: 0.9 },
   { input: [5, 5, 2], target: 0.5 },
   { input: [2, 1, 1], target: 0.2 },
   { input: [6, 7, 2], target: 0.7 }
 ];

 // =============== Poids & biais initiaux ===============
 // Node1 => i1,i2 | Node2 => i2,i3 | Node3 => i1,i2,i3
 let w_input_h1 = [
   // input1 => [h1_1, h1_2, h1_3]
   [0.1, 0.0, 0.3],  
   // input2 => [h1_1, h1_2, h1_3]
   [0.2, 0.4, 0.5],  
   // input3 => [h1_1, h1_2, h1_3]
   [0.0, 0.6, 0.7]   
 ];
 let b_h1 = [0.1, 0.1, 0.1];

 let w_h1_out = [
   [0.2], // h1_1 -> out
   [0.3], // h1_2 -> out
   [0.4]  // h1_3 -> out
 ];
 let b_out = [0.1];

 function sigmoid(x) {
   return 1 / (1 + Math.exp(-x));
 }
 function dsigmoid(y) {
   return y * (1 - y);
 }

 // =============== Forward pass ===============
 function forwardPass(inputs) {
   const [i1, i2, i3] = inputs;

   const h1_1_in = i1 * w_input_h1[0][0]
                 + i2 * w_input_h1[1][0]
                 + i3 * w_input_h1[2][0]
                 + b_h1[0];
   const h1_1_out = sigmoid(h1_1_in);

   const h1_2_in = i1 * w_input_h1[0][1]
                 + i2 * w_input_h1[1][1]
                 + i3 * w_input_h1[2][1]
                 + b_h1[1];
   const h1_2_out = sigmoid(h1_2_in);

   const h1_3_in = i1 * w_input_h1[0][2]
                 + i2 * w_input_h1[1][2]
                 + i3 * w_input_h1[2][2]
                 + b_h1[2];
   const h1_3_out = sigmoid(h1_3_in);

   const out_in = (h1_1_out * w_h1_out[0][0]) 
                + (h1_2_out * w_h1_out[1][0])
                + (h1_3_out * w_h1_out[2][0])
                + b_out[0];
   const out = sigmoid(out_in);

   return { h1_1_out, h1_2_out, h1_3_out, out };
 }

 // =============== Backprop ===============
 function backwardPass(inputs, forwardResults, target) {
   const { h1_1_out, h1_2_out, h1_3_out, out } = forwardResults;
   const errorOut = target - out;
   const gradOut = errorOut * dsigmoid(out);

   const old_h1_1_out = w_h1_out[0][0];
   const old_h1_2_out = w_h1_out[1][0];
   const old_h1_3_out = w_h1_out[2][0];

   // update hidden->out
   w_h1_out[0][0] += learningRate * gradOut * h1_1_out;
   w_h1_out[1][0] += learningRate * gradOut * h1_2_out;
   w_h1_out[2][0] += learningRate * gradOut * h1_3_out;
   b_out[0]       += learningRate * gradOut;

   const error_h1_1 = gradOut * old_h1_1_out;
   const error_h1_2 = gradOut * old_h1_2_out;
   const error_h1_3 = gradOut * old_h1_3_out;

   const grad_h1_1 = error_h1_1 * dsigmoid(h1_1_out);
   const grad_h1_2 = error_h1_2 * dsigmoid(h1_2_out);
   const grad_h1_3 = error_h1_3 * dsigmoid(h1_3_out);

   const [i1, i2, i3] = inputs;

   // node1
   w_input_h1[0][0] += learningRate * grad_h1_1 * i1; 
   w_input_h1[1][0] += learningRate * grad_h1_1 * i2; 
   w_input_h1[2][0] += learningRate * grad_h1_1 * i3; 

   // node2
   w_input_h1[0][1] += learningRate * grad_h1_2 * i1; 
   w_input_h1[1][1] += learningRate * grad_h1_2 * i2;
   w_input_h1[2][1] += learningRate * grad_h1_2 * i3;

   // node3
   w_input_h1[0][2] += learningRate * grad_h1_3 * i1; 
   w_input_h1[1][2] += learningRate * grad_h1_3 * i2;
   w_input_h1[2][2] += learningRate * grad_h1_3 * i3;

   b_h1[0] += learningRate * grad_h1_1;
   b_h1[1] += learningRate * grad_h1_2;
   b_h1[2] += learningRate * grad_h1_3;
 }

 // =============== Entraînement pas à pas ===============
 function trainOneStep() {
   if (currentEpoch > maxEpochs) {
     document.getElementById('explanation').innerHTML =
       `Entraînement terminé: ${maxEpochs} époques accomplies.`;
     return;
   }
   const sample = trainingData[currentSampleIndex];
   const fwd = forwardPass(sample.input);
   backwardPass(sample.input, fwd, sample.target);

   displayState(sample.input, fwd.out, sample.target);

   currentSampleIndex++;
   if (currentSampleIndex >= trainingData.length) {
     currentSampleIndex = 0;
     currentEpoch++;
   }
 }

 // "Sauter 10% d'époques" => each step is 1 sample
 // so 10% epochs => #epochs * trainingData.length => total steps
 function skipTenPercent() {
   const stepsToDo = trainingData.length * Math.floor(maxEpochs * 0.1); 
   const initialEpoch = currentEpoch;
   for (let i = 0; i < stepsToDo; i++) {
     if (currentEpoch > maxEpochs) break;
     trainOneStep();
   }
   if (currentEpoch > maxEpochs) {
     document.getElementById('explanation').innerHTML =
       `Nous avons terminé à l&apos;epoch #${maxEpochs}.`;
   } else {
     document.getElementById('explanation').innerHTML += 
       `<br>Nous avons avancé de ${stepsToDo} étapes (de l&apos;epoch ${initialEpoch} à ${currentEpoch}).`;
   }
 }

 // Train all epochs at once
 function trainAll() {
   while (currentEpoch <= maxEpochs) {
     trainOneStep();
   }
 }

 function displayState(inputs, output, target) {
   document.querySelector('[data-name="input1"] .neuron-value').textContent = inputs[0];
   document.querySelector('[data-name="input2"] .neuron-value').textContent = inputs[1];
   document.querySelector('[data-name="input3"] .neuron-value').textContent = inputs[2];

   const { h1_1_out, h1_2_out, h1_3_out, out } = forwardPass(inputs);

   document.querySelector('[data-name="h1_1"] .neuron-value').textContent = h1_1_out.toFixed(2);
   document.querySelector('[data-name="h1_2"] .neuron-value').textContent = h1_2_out.toFixed(2);
   document.querySelector('[data-name="h1_3"] .neuron-value').textContent = h1_3_out.toFixed(2);
   document.querySelector('[data-name="output"] .neuron-value').textContent = out.toFixed(2);

   document.getElementById('bias-h1_1').textContent = "b=" + b_h1[0].toFixed(2);
   document.getElementById('bias-h1_2').textContent = "b=" + b_h1[1].toFixed(2);
   document.getElementById('bias-h1_3').textContent = "b=" + b_h1[2].toFixed(2);
   document.getElementById('bias-out').textContent  = "b=" + b_out[0].toFixed(2);

   document.getElementById('w_00_text').textContent = w_input_h1[0][0].toFixed(2); 
   document.getElementById('w_10_text').textContent = w_input_h1[1][0].toFixed(2);
   document.getElementById('w_20_text').textContent = w_input_h1[2][0].toFixed(2);

   document.getElementById('w_01_text').textContent = w_input_h1[0][1].toFixed(2);
   document.getElementById('w_11_text').textContent = w_input_h1[1][1].toFixed(2);
   document.getElementById('w_21_text').textContent = w_input_h1[2][1].toFixed(2);

   document.getElementById('w_02_text').textContent = w_input_h1[0][2].toFixed(2);
   document.getElementById('w_12_text').textContent = w_input_h1[1][2].toFixed(2);
   document.getElementById('w_22_text').textContent = w_input_h1[2][2].toFixed(2);

   document.getElementById('w_h1_1_out').textContent = w_h1_out[0][0].toFixed(2);
   document.getElementById('w_h1_2_out').textContent = w_h1_out[1][0].toFixed(2);
   document.getElementById('w_h1_3_out').textContent = w_h1_out[2][0].toFixed(2);

   document.getElementById('explanation').innerHTML = 
     `<strong>Epoch ${currentEpoch}/${maxEpochs}</strong> – Sample #${currentSampleIndex+1}<br>` +
     `Entrée : [${inputs}] – Cible : ${target}<br>` +
     `Sortie : ${output.toFixed(2)} – Erreur : ${(target - output).toFixed(2)}`;

   // Also update the "copyable" code block
   updateCopyableWeights();
 }

 // Creates or updates a text snippet that we can copy/paste
 function updateCopyableWeights() {
   let snippet = 
`w_input_h1 = [
[${w_input_h1[0][0].toFixed(2)}, ${w_input_h1[0][1].toFixed(2)}, ${w_input_h1[0][2].toFixed(2)}],
[${w_input_h1[1][0].toFixed(2)}, ${w_input_h1[1][1].toFixed(2)}, ${w_input_h1[1][2].toFixed(2)}],
[${w_input_h1[2][0].toFixed(2)}, ${w_input_h1[2][1].toFixed(2)}, ${w_input_h1[2][2].toFixed(2)}]
];
b_h1 = [${b_h1[0].toFixed(2)}, ${b_h1[1].toFixed(2)}, ${b_h1[2].toFixed(2)}];
w_h1_out = [
[${w_h1_out[0][0].toFixed(2)}],
[${w_h1_out[1][0].toFixed(2)}],
[${w_h1_out[2][0].toFixed(2)}]
];
b_out = [${b_out[0].toFixed(2)}];`;

   document.getElementById('copyable-weights').textContent = snippet;
 }

 // =============== Réinitialiser ===============
 function reinitialiser() {
   currentEpoch = 1;
   currentSampleIndex = 0;

   // Reinit with same default
   w_input_h1 = [
     [0.1, 0.0, 0.3],
     [0.2, 0.4, 0.5],
     [0.0, 0.6, 0.7]
   ];
   b_h1 = [0.1, 0.1, 0.1];
   w_h1_out = [
     [0.2],
     [0.3],
     [0.4]
   ];
   b_out = [0.1];

   // Clear UI
   document.querySelector('[data-name="input1"] .neuron-value').textContent = 0;
   document.querySelector('[data-name="input2"] .neuron-value').textContent = 0;
   document.querySelector('[data-name="input3"] .neuron-value').textContent = 0;

   document.querySelector('[data-name="h1_1"] .neuron-value').textContent = '...';
   document.querySelector('[data-name="h1_2"] .neuron-value').textContent = '...';
   document.querySelector('[data-name="h1_3"] .neuron-value').textContent = '...';
   document.querySelector('[data-name="output"] .neuron-value').textContent = '...';

   document.getElementById('bias-h1_1').textContent = "b=0.10";
   document.getElementById('bias-h1_2').textContent = "b=0.10";
   document.getElementById('bias-h1_3').textContent = "b=0.10";
   document.getElementById('bias-out').textContent  = "b=0.10";

   document.getElementById('w_00_text').textContent = "";
   document.getElementById('w_10_text').textContent = "";
   document.getElementById('w_20_text').textContent = "";
   document.getElementById('w_01_text').textContent = "";
   document.getElementById('w_11_text').textContent = "";
   document.getElementById('w_21_text').textContent = "";
   document.getElementById('w_02_text').textContent = "";
   document.getElementById('w_12_text').textContent = "";
   document.getElementById('w_22_text').textContent = "";
   document.getElementById('w_h1_1_out').textContent = "";
   document.getElementById('w_h1_2_out').textContent = "";
   document.getElementById('w_h1_3_out').textContent = "";

   document.getElementById('explanation').innerHTML =
     '<strong>Explication :</strong><br>' +
     '• Node1=Action+Comedy, Node2=Comedy+Duration, Node3=All.<br>' +
     '• maxEpochs=200, learningRate=0.5.<br>' +
     'Cliquez sur « Étape suivante » / « Sauter 10% » / « Tout entraîner ».';
   
   // Update the copyable snippet for newly reset defaults
   updateCopyableWeights();
 }

 // =============== Dessin des connexions ===============
 window.addEventListener('load', () => {
   dessinerConnexions();
   updateCopyableWeights(); // Show default at the start
 });
 window.addEventListener('resize', dessinerConnexions);

 function dessinerConnexions() {
   document.querySelectorAll('.connection').forEach(el => el.remove());
   const net = document.getElementById('network');
   const neurons = document.querySelectorAll('.neuron');

   neurons.forEach(fromNeuron => {
     const fromName = fromNeuron.getAttribute('data-name');
     const fromRect = fromNeuron.getBoundingClientRect();

     neurons.forEach(toNeuron => {
       const toName = toNeuron.getAttribute('data-name');
       if (!shouldConnect(fromName, toName)) return;

       const toRect = toNeuron.getBoundingClientRect();
       const line = document.createElement('div');
       line.className = 'connection';

       line.setAttribute('data-from', fromName);
       line.setAttribute('data-to', toName);

       const length = Math.sqrt(
         Math.pow(toRect.left - fromRect.right, 2) +
         Math.pow(toRect.top - fromRect.top, 2)
       );
       const angle = Math.atan2(
         toRect.top - fromRect.top,
         toRect.left - fromRect.right
       );
       
       line.style.width = `${length}px`;
       line.style.transform = `rotate(${angle}rad)`;
       line.style.left = `${fromRect.right - fromRect.width/2}px`;
       line.style.top = `${fromRect.top + fromRect.height/2}px`;

       net.appendChild(line);
     });
   });
 }

 function shouldConnect(fromName, toName) {
   const inputNames = ["input1", "input2", "input3"];
   const hiddenNames = ["h1_1", "h1_2", "h1_3"];
   if (inputNames.includes(fromName) && hiddenNames.includes(toName)) {
     return true;
   }
   if (hiddenNames.includes(fromName) && toName === "output") {
     return true;
   }
   return false;
 }