const socket = typeof io === 'function' ? io({
  transports: ['websocket', 'polling'],
  reconnection: true,
  reconnectionDelay: 1000
}) : null;

const catLabel = {
  red: 'Resusitasi',
  yellow: 'Darurat',
  green: 'Non-Darurat'
};

let bedsData = {};
let selectedBedId = null;

async function fetchInitialBeds() {
  try {
    const response = await fetch('/api/beds');
    if (response.ok) {
      bedsData = await response.json();
      renderAllZones();
    }
  } catch (error) {
    console.error("[ERROR] Gagal mengambil data bed awal:", error);
  }
}

function renderAllZones() {
  renderZone('zoneA', ['A1', 'A2', 'A3', 'A4', 'A5'], 0);
  renderZone('zoneB', ['B1', 'B2', 'B3', 'B4'], 10);
  renderZone('zoneC', ['C1', 'C2', 'C3'], 60);
  updateStats();
  
  if (selectedBedId && bedsData[selectedBedId]) {
    displayBedDetail(selectedBedId);
  }
}

function renderZone(elementId, bedIds, targetMinutes) {
  const container = document.getElementById(elementId);
  if (!container) return;
  
  container.innerHTML = '';

  bedIds.forEach(id => {
    const bedInfo = bedsData[id] || { status: 'empty' };
    const cat = bedInfo.status !== 'empty' ? bedInfo.status : null;

    const bed = document.createElement('div');
    bed.className = 'bed' + (cat ? ' ' + cat : ' empty');
    bed.dataset.cat = cat || '';
    bed.dataset.id = id;
    bed.dataset.target = targetMinutes;

    if (cat && bedInfo.arrival_timestamp) {
      bed.dataset.arrival = bedInfo.arrival_timestamp;
    }

    bed.innerHTML = `<div class="bed-id mono">${id}</div>` +
      (cat ? `<div class="bed-cat">${catLabel[cat]}</div><div class="bed-timer mono" data-timer></div>` : '');

    // KETIKA BOX BED DIKLIK: TAMPILKAN DI PANEL SISI KANAN
    bed.onclick = () => {
      selectedBedId = id;
      displayBedDetail(id);
    };
    
    container.appendChild(bed);
  });
}

// FUNGSI UTAMA: MENAMPILKAN DETAIL BED DI PANEL KANAN
function displayBedDetail(bedId) {
  const bedInfo = bedsData[bedId] || { status: 'empty' };
  const titleEl = document.getElementById('panelBedTitle');
  const contentEl = document.getElementById('bedDetailContent');

  titleEl.textContent = `DETAIL BED — ${bedId}`;

  if (bedInfo.status === 'empty') {
    contentEl.innerHTML = `
      <div class="modal-row" style="border-color:var(--border)"><span>Status Bed</span><span class="mono">KOSONG / TERSEDIA</span></div>
      <p style="color: var(--text-muted); font-size: 11px; margin-top: 10px;">Bed ini siap digunakan untuk pasien baru.</p>
    `;
    return;
  }

  const vitals = bedInfo.vitals || {};
  const arrivalTime = bedInfo.arrival_timestamp ? new Date(bedInfo.arrival_timestamp).toLocaleTimeString('id-ID') : '—';

  contentEl.innerHTML = `
    <div class="modal-row" style="border-color:var(--border)"><span>Status Triase</span><span style="font-weight:bold; color:var(--${bedInfo.status})">${catLabel[bedInfo.status]}</span></div>
    <div class="modal-row" style="border-color:var(--border)"><span>Skor XGBoost</span><span class="mono">${bedInfo.xgboost_score || '—'}</span></div>
    <div class="modal-row" style="border-color:var(--border)"><span>Skor GCS (Hardware)</span><span class="mono" style="font-weight:bold;">${bedInfo.gcs_score || 15} / 15</span></div>
    <div class="modal-row" style="border-color:var(--border)"><span>Waktu Masuk</span><span class="mono">${arrivalTime}</span></div>
    
    <div style="margin-top: 12px; font-size: 11px; color: var(--accent); font-weight: bold; text-transform: uppercase;">Tanda Vital (Hasil Cek Hardware)</div>
    <div class="modal-row" style="border-color:var(--border)"><span>Heart Rate</span><span class="mono">${vitals.hr || '—'} BPM</span></div>
    <div class="modal-row" style="border-color:var(--border)"><span>SpO2</span><span class="mono">${vitals.spo2 || '—'} %</span></div>
    <div class="modal-row" style="border-color:var(--border)"><span>Resp. Rate</span><span class="mono">${vitals.rr || '—'} RPM</span></div>
    <div class="modal-row" style="border-color:var(--border)"><span>Tekanan Darah</span><span class="mono">${(vitals.sys && vitals.dia) ? `${vitals.sys}/${vitals.dia}` : '—'} mmHg</span></div>
    <div class="modal-row" style="border-color:var(--border)"><span>Suhu Tubuh Inti</span><span class="mono" style="color:var(--accent);">${vitals.temp_core || '—'} °C</span></div>

    <div style="margin-top: 12px; font-size: 11px; color: var(--text-secondary); font-weight: bold; text-transform: uppercase;">Identitas Pasien</div>
    <div class="modal-row" style="border-color:var(--border)"><span>Nama Pasien</span><span class="mono">${bedInfo.patient_name || '<i>Belum Diisi</i>'}</span></div>
    <div class="modal-row" style="border-color:var(--border)"><span>Penanggung Jawab</span><span class="mono">${bedInfo.relative_name || '<i>Belum Diisi</i>'}</span></div>

    <button onclick="openFormModal('${bedId}')" style="width:100%; margin-top:15px; padding:8px; background:var(--accent); color:#fff; font-weight:bold; border:none; border-radius:6px; cursor:pointer;">
      + Lengkapi Data Pasien
    </button>
    <button onclick="dischargeBed('${bedId}')" style="width:100%; margin-top:8px; padding:8px; background:var(--red); color:#fff; font-weight:bold; border:none; border-radius:6px; cursor:pointer;">
      Kosongkan Bed
    </button>
  `;
}

async function dischargeBed(bedId) {
  try {
    const res = await fetch(`/api/beds/${bedId}/discharge`, { method: 'POST' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    // Panel detail otomatis terupdate lewat event 'bed_updated' dari socket.
  } catch (error) {
    console.error('[ERROR] Gagal mengosongkan bed:', error);
    alert('Gagal mengosongkan bed. Coba lagi.');
  }
}

// MODAL REGISTRASI PASIEN
function openFormModal(bedId) {
  document.getElementById('formBedId').textContent = bedId;
  const bedInfo = bedsData[bedId] || {};
  document.getElementById('inputPatientName').value = bedInfo.patient_name || '';
  document.getElementById('patientFormModal').classList.add('show');
}

function closeFormModal() {
  document.getElementById('patientFormModal').classList.remove('show');
}

async function savePatientData() {
  const bedId = document.getElementById('formBedId').textContent;
  const patientName = document.getElementById('inputPatientName').value.trim();
  const relativeName = document.getElementById('inputRelativeName').value.trim();

  try {
    const response = await fetch(`/api/beds/${bedId}/patient`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        patient_name: patientName,
        relative_name: relativeName
      })
    });

    const result = await response.json();
    if (response.ok && result.status === 'success') {
      if (bedsData[bedId]) {
        bedsData[bedId].patient_name = patientName || 'Pasien Tanpa Nama';
        bedsData[bedId].relative_name = relativeName || '—';
        displayBedDetail(bedId);
      }
      closeFormModal();
    } else {
      alert('Gagal menyimpan data pasien: ' + (result.message || 'Error tidak diketahui'));
    }
  } catch (err) {
    console.error('[ERROR] Gagal menyimpan data pasien ke server:', err);
    alert('Gagal menyimpan data pasien. Pastikan koneksi server backend terhubung!');
  }
}

if (socket) {
  socket.on('bed_updated', (updatedBed) => {
    bedsData[updatedBed.bed_id] = updatedBed;
    renderAllZones();
    tickCountdowns();
  });

  socket.on('beds_reset', (allBeds) => {
    bedsData = allBeds;
    renderAllZones();
    tickCountdowns();
  });
} else {
  console.warn('[SOCKET] Socket.IO tidak tersedia; menggunakan polling API.');
  setInterval(fetchInitialBeds, 3000);
}

function tickCountdowns() {
  document.querySelectorAll('.bed[data-arrival]').forEach(bed => {
    const target = parseFloat(bed.dataset.target);
    const arrival = parseFloat(bed.dataset.arrival);
    const elapsedSec = (Date.now() - arrival) / 1000;
    const remainingSec = target * 60 - elapsedSec;
    const timerEl = bed.querySelector('[data-timer]');
    
    if (!timerEl) return;

    const sign = remainingSec < 0 ? '-' : '';
    const abs = Math.abs(Math.round(remainingSec));
    const mm = String(Math.floor(abs / 60)).padStart(2, '0');
    const ss = String(abs % 60).padStart(2, '0');
    
    timerEl.textContent = sign + mm + ':' + ss;
    bed.classList.toggle('overdue', remainingSec < 0);
  });
}

function updateStats() {
  const beds = document.querySelectorAll('.bed');
  const counts = { red: 0, yellow: 0, green: 0 };
  let occupied = 0;
  beds.forEach(b => {
    if (b.dataset.cat) { counts[b.dataset.cat]++; occupied++; }
  });
  document.getElementById('countRed').textContent = counts.red;
  document.getElementById('countYellow').textContent = counts.yellow;
  document.getElementById('countGreen').textContent = counts.green;
  document.getElementById('occupied').textContent = occupied + ' / ' + beds.length;
  document.getElementById('available').textContent = (beds.length - occupied) + ' bed';
}

function tickClock() {
  const now = new Date();
  document.getElementById('clock').textContent = now.toLocaleTimeString('id-ID');
  document.getElementById('clockDate').textContent = now.toLocaleDateString('id-ID', { weekday: 'long', day: 'numeric', month: 'long' });
}

document.addEventListener('DOMContentLoaded', () => {
  fetchInitialBeds();
  tickClock();
  setInterval(tickClock, 1000);
  setInterval(tickCountdowns, 1000);

  // REAL-TIME SOCKET.IO LISTENERS FOR INSTANT BED UPDATES WITHOUT REFRESH
  if (socket) {
    socket.on('connect', () => {
      console.log('[SOCKET] Terhubung ke TriaGO Real-Time Server.');
    });

    socket.on('bed_update', handleSingleBedUpdate);
    socket.on('bed_updated', handleSingleBedUpdate);

    function handleSingleBedUpdate(bedInfo) {
      console.log('[SOCKET] Real-time Bed Update:', bedInfo);
      if (bedInfo && bedInfo.bed_id) {
        bedsData[bedInfo.bed_id] = bedInfo;
        renderAllZones();
      }
    }

    socket.on('beds_matrix_update', (allBeds) => {
      console.log('[SOCKET] Real-time Matrix Update:', allBeds);
      if (allBeds && typeof allBeds === 'object') {
        bedsData = allBeds;
        renderAllZones();
      }
    });
  }
});