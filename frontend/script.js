// Hamburger & mobile menu
const hamburger = document.querySelector('.hamburger');
const mobileMenu = document.querySelector('.mobile-menu');

hamburger.addEventListener('click', () => {
  mobileMenu.classList.toggle('hidden');
  hamburger.classList.toggle('open'); // optional: add .open class for X animation
});

// Page switching logic (same as before, just cleaner)
const sections = {
  home: document.getElementById('home'),
  about: document.getElementById('about'),
  dataset: document.getElementById('dataset'),
  prototype: document.getElementById('prototype')
};

const backBtn = document.getElementById('backBtn');

function showSection(sectionId) {
  document.querySelectorAll('.page-section').forEach(el => {
    el.classList.remove('active');
  });
  sections[sectionId].classList.add('active');

  backBtn.classList.toggle('hidden', sectionId === 'home');
  backBtn.classList.toggle('flex', sectionId !== 'home');

  mobileMenu.classList.add('hidden'); // close mobile on click
}

document.querySelectorAll('[data-section]').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    showSection(link.dataset.section);
  });
});

backBtn.addEventListener('click', () => showSection('home'));

// Start
showSection('home');

// Model population + prediction (your accuracies)
const modelData = {
  "DistilBERT": 94.43,
  "BERTimbau": 88.92,
};

const modelSelect = document.getElementById('modelSelect');

Object.entries(modelData).forEach(([name, acc]) => {
  const opt = document.createElement('option');
  opt.value = name;
  opt.textContent = `${name} (${acc.toFixed(2)}%)`;
  modelSelect.appendChild(opt);
});

const predictBtn = document.getElementById('predictBtn');
const resultDiv = document.getElementById('result');

predictBtn.addEventListener('click', async () => {
  const text = document.getElementById('textInput').value.trim();
  const model = modelSelect.value;

  if (!text || !model) {
    alert('Please enter text and select a model.');
    return;
  }

  resultDiv.innerHTML = '<span class="text-slate-500">Analyzing...</span>';
  resultDiv.className = 'mt-8 p-6 rounded-xl min-h-[120px] text-center text-xl font-medium bg-slate-100';

  try {
    const res = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, model })
    });

    const data = await res.json();

    if (data.error) throw new Error(data.error);

    const isSuicidal = data.prediction === 'Suicidal';
    resultDiv.innerHTML = `
      <div class="text-2xl font-bold mb-3 ${isSuicidal ? 'text-rose-700' : 'text-emerald-700'}">
        ${isSuicidal ? 'Potential Risk Detected' : 'Low Risk Indication'}
      </div>
      <div class="text-lg">
        Confidence: <strong>${data.confidence.toFixed(1)}%</strong><br>
        Model accuracy: <strong>${modelData[model]}%</strong>
      </div>
    `;
    resultDiv.className += isSuicidal
      ? ' bg-rose-50 border border-rose-200 text-rose-900'
      : ' bg-emerald-50 border border-emerald-200 text-emerald-900';
  } catch (err) {
    resultDiv.innerHTML = 'Error: Could not connect or invalid response.';
    resultDiv.className += ' bg-rose-50 border border-rose-200 text-rose-900';
  }
});