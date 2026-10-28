// Semua URL relatif terhadap folder halaman, agar dashboard tetap jalan di
// balik reverse proxy sub-path (mis. /informatics/triago/).
const BASE_PATH = (typeof location !== 'undefined' && location.pathname)
  ? location.pathname.replace(/[^/]*$/, '') : '/';
const socket = typeof io === 'function' ? io({ path: `${BASE_PATH}socket.io` }) : null;

const catLabel = {
  red: 'Resusitasi',
  yellow: 'Darurat',
  green: 'Non-Darurat'
};

// Denah bentuk U, sisi atas terbuka ke pintu masuk & area triase:
// sisi kiri = zona Resusitasi, sisi bawah = Darurat, sisi kanan = Non-Darurat.
// Ukuran denah mengikuti jumlah bed per zona yang dikirim backend.
function floorLayout(beds) {
  const ids = zone => Object.keys(beds).sort().filter(id => beds[id].zone === zone);
  const [left, bottom, right] = [ids('red'), ids('yellow'), ids('green')];
  const cols = Math.max(bottom.length, 3);
  const rows = Math.max(left.length, right.length, 2) + 1;
  const offset = 1 + Math.floor((cols - bottom.length) / 2);
  const positions = {};
  left.forEach((id, i) => { positions[id] = [i + 1, 1]; });
  bottom.forEach((id, i) => { positions[id] = [rows, offset + i]; });
  // Nomor berlanjut dari pojok kanan bawah ke atas, mengikuti alur huruf U.
  right.forEach((id, i) => { positions[id] = [rows - 1 - i, cols]; });
  return { cols, rows, positions };
}

// Field registrasi pasien: [key, label, opsi pilihan (jika select)].
const DETAIL_SECTIONS = [
  ['Identitas Pasien', [
    ['medical_record_no', 'No. Rekam Medis'], ['nik', 'NIK'], ['birth_date', 'Tanggal Lahir'],
    ['sex', 'Jenis Kelamin', { L: 'Laki-laki', P: 'Perempuan' }],
    ['address', 'Alamat'], ['phone', 'No. HP']
  ]],
  ['Kedatangan & Penjamin', [
    ['arrival_mode', 'Cara Datang', { sendiri: 'Datang sendiri', diantar: 'Diantar', ambulans: 'Ambulans', rujukan: 'Rujukan' }],
    ['payer', 'Penjamin', { umum: 'Umum', bpjs: 'BPJS Kesehatan', asuransi: 'Asuransi lain' }],
    ['insurance_no', 'No. Kartu']
  ]],
  ['Klinis', [['chief_complaint', 'Keluhan Utama'], ['allergies', 'Riwayat Alergi']]],
  ['Penanggung Jawab', [['relative_relation', 'Hubungan'], ['relative_phone', 'No. HP']]]
];
const DETAIL_KEYS = DETAIL_SECTIONS.flatMap(([, fields]) => fields.map(([key]) => key));

let bedsData = {};
let selectedBedId = null;

function escapeHTML(value) {
  const entities = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  return String(value).replace(/[&<>"']/g, character => entities[character]);
}

async function fetchInitialBeds() {
  try {
    const response = await fetch('api/beds');
    if (response.ok) {
      bedsData = await response.json();
      renderAllZones();
    }
  } catch (error) {
    console.error("[ERROR] Gagal mengambil data bed awal:", error);
  }
}

function renderAllZones() {
  renderFloor();
  updateStats();

  if (selectedBedId && bedsData[selectedBedId]) {
    displayBedDetail(selectedBedId);
  }
}

function renderFloor() {
  const container = document.getElementById('igdFloor');
  if (!container) return;

  container.querySelectorAll('.bed').forEach(bed => bed.remove());

  const layout = floorLayout(bedsData);
  container.style.gridTemplateColumns = `repeat(${layout.cols},minmax(0,1fr))`;
  container.style.gridTemplateRows = `repeat(${layout.rows},auto)`;
  const inner = `2 / ${layout.cols}`;
  container.querySelector('.floor-entrance').style.gridColumn = inner;
  const station = container.querySelector('.nurse-station');
  station.style.gridColumn = inner;
  station.style.gridRow = `2 / ${layout.rows}`;

  Object.keys(bedsData).sort().forEach(id => {
    const bedInfo = bedsData[id] || { status: 'empty' };
    const cat = bedInfo.status !== 'empty' ? bedInfo.status : null;

    const bed = document.createElement('div');
    bed.className = 'bed' + (cat ? ' ' + cat : ' empty') + ' zone-' + bedInfo.zone +
      (id === selectedBedId ? ' selected' : '');
    bed.dataset.cat = cat || '';
    bed.dataset.id = id;
    bed.dataset.target = bedInfo.target_minutes ?? 0;
    const position = layout.positions[id];
    if (position) {
      bed.style.gridRow = position[0];
      bed.style.gridColumn = position[1];
    }

    if (cat && bedInfo.arrival_timestamp) {
      bed.dataset.arrival = bedInfo.arrival_timestamp;
    }

    bed.innerHTML = `<div class="bed-id mono">${escapeHTML(id)}</div>` +
      (cat ? `<div class="bed-cat">${catLabel[cat]}</div><div class="bed-timer mono" data-timer></div>`
           : `<div class="bed-zone">${catLabel[bedInfo.zone] || ''}</div>`);

    // KETIKA BOX BED DIKLIK: TAMPILKAN DI PANEL SISI KANAN
    bed.onclick = () => {
      selectedBedId = id;
      renderAllZones();
      tickCountdowns();
    };

    container.appendChild(bed);
  });
}

function formatDetail(key, value, options) {
  if (options) return options[value] || value;
  if (key === 'birth_date') {
    const born = new Date(value + 'T00:00:00');
    if (isNaN(born)) return value;
    const now = new Date();
    let age = now.getFullYear() - born.getFullYear();
    if (now < new Date(now.getFullYear(), born.getMonth(), born.getDate())) age--;
    return `${born.toLocaleDateString('id-ID')} (${age} th)`;
  }
  return value;
}

function renderPatientDetails(bedInfo) {
  const details = bedInfo.patient_details || {};
  const row = (label, value) =>
    `<div class="modal-row" style="border-color:var(--border)"><span>${label}</span><span class="mono detail-value">${value}</span></div>`;
  const empty = '<i>Belum Diisi</i>';

  return DETAIL_SECTIONS.map(([title, fields]) => {
    const rows = [];
    if (title === 'Identitas Pasien') {
      rows.push(row('Nama Pasien', bedInfo.patient_name ? escapeHTML(bedInfo.patient_name) : empty));
    }
    if (title === 'Penanggung Jawab') {
      rows.push(row('Nama', bedInfo.relative_name ? escapeHTML(bedInfo.relative_name) : empty));
    }
    fields.forEach(([key, label, options]) => {
      if (details[key]) rows.push(row(label, escapeHTML(formatDetail(key, details[key], options))));
    });
    if (!rows.length) return '';
    return `<div class="detail-section">${title}</div>` + rows.join('');
  }).join('');
}

// FUNGSI UTAMA: MENAMPILKAN DETAIL BED DI PANEL KANAN
function displayBedDetail(bedId) {
  const bedInfo = bedsData[bedId] || { status: 'empty' };
  const titleEl = document.getElementById('panelBedTitle');
  const contentEl = document.getElementById('bedDetailContent');

  titleEl.textContent = `DETAIL BED — ${bedId}`;
  const zoneRow = `<div class="modal-row" style="border-color:var(--border)"><span>Zona Bed</span><span>${catLabel[bedInfo.zone] || '—'}</span></div>`;

  if (bedInfo.status === 'empty') {
    contentEl.innerHTML = zoneRow + `
      <div class="modal-row" style="border-color:var(--border)"><span>Status Bed</span><span class="mono">KOSONG / TERSEDIA</span></div>
      <p style="color: var(--text-muted); font-size: 11px; margin-top: 10px;">Bed ini siap digunakan untuk pasien baru.</p>
    `;
    return;
  }

  const vitals = bedInfo.vitals || {};
  const arrivalTime = bedInfo.arrival_timestamp ? new Date(bedInfo.arrival_timestamp).toLocaleTimeString('id-ID') : '—';

  contentEl.innerHTML = zoneRow + `
    <div class="modal-row" style="border-color:var(--border)"><span>Status Triase</span><span style="font-weight:bold; color:var(--${bedInfo.status})">${catLabel[bedInfo.status]}</span></div>
    <div class="modal-row" style="border-color:var(--border)"><span>Skor XGBoost</span><span class="mono">${bedInfo.xgboost_score || '—'}</span></div>
    <div class="modal-row" style="border-color:var(--border)"><span>Skor GCS</span><span class="mono" style="font-weight:bold;">${bedInfo.gcs_score || 15} / 15</span></div>
    <div class="modal-row" style="border-color:var(--border)"><span>Pain Score</span><span class="mono">${bedInfo.pain_score == null ? '—' : escapeHTML(bedInfo.pain_score) + ' / 10'}</span></div>
    <div class="modal-row" style="border-color:var(--border)"><span>Waktu Masuk</span><span class="mono">${arrivalTime}</span></div>
    
    <div style="margin-top: 12px; font-size: 11px; color: var(--accent); font-weight: bold; text-transform: uppercase;">Tanda Vital (Hasil Cek Hardware)</div>
    <div class="modal-row" style="border-color:var(--border)"><span>Heart Rate</span><span class="mono">${vitals.hr || '—'} BPM</span></div>
    <div class="modal-row" style="border-color:var(--border)"><span>SpO2</span><span class="mono">${vitals.spo2 || '—'} %</span></div>
    <div class="modal-row" style="border-color:var(--border)"><span>Resp. Rate</span><span class="mono">${vitals.rr || '—'} RPM</span></div>
    <div class="modal-row" style="border-color:var(--border)"><span>Tekanan Darah</span><span class="mono">${(vitals.sys && vitals.dia) ? `${vitals.sys}/${vitals.dia}` : '—'} mmHg</span></div>
    <div class="modal-row" style="border-color:var(--border)"><span>Suhu Tubuh Inti</span><span class="mono" style="color:var(--accent);">${vitals.temp_core || '—'} °C</span></div>

    ${renderPatientDetails(bedInfo)}

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
    if (!confirm(`Kosongkan Bed ${bedId}? Data pasien di bed ini akan dihapus dari dashboard.`)) return;
    const res = await fetch(`api/beds/${encodeURIComponent(bedId)}/discharge`, { method: 'POST' });
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
  const details = bedInfo.patient_details || {};
  document.getElementById('inputPatientName').value = bedInfo.patient_name || '';
  document.getElementById('inputRelativeName').value =
    bedInfo.relative_name && bedInfo.relative_name !== '—' ? bedInfo.relative_name : '';
  DETAIL_KEYS.forEach(key => { document.getElementById('f_' + key).value = details[key] || ''; });
  document.getElementById('f_birth_date').max = new Date().toISOString().slice(0, 10);
  document.getElementById('formError').textContent = '';
  document.getElementById('patientFormModal').classList.add('show');
  document.getElementById('inputPatientName').focus();
}

function closeFormModal() {
  document.getElementById('patientFormModal').classList.remove('show');
}

async function savePatientData(event) {
  if (event) event.preventDefault();
  const bedId = document.getElementById('formBedId').textContent;
  const errorEl = document.getElementById('formError');
  const details = {};
  DETAIL_KEYS.forEach(key => { details[key] = document.getElementById('f_' + key).value.trim(); });

  try {
    const response = await fetch(`api/beds/${encodeURIComponent(bedId)}/patient`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        patient_name: document.getElementById('inputPatientName').value.trim(),
        relative_name: document.getElementById('inputRelativeName').value.trim(),
        details
      })
    });

    const result = await response.json();
    if (response.ok && result.status === 'success') {
      bedsData[bedId] = result.data;
      renderAllZones();
      tickCountdowns();
      closeFormModal();
      if (!result.saved_to_database) {
        alert('Data tampil di dashboard, tetapi belum tersimpan ke database. Periksa koneksi database.');
      }
    } else {
      errorEl.textContent = 'Gagal menyimpan: ' + (result.message || 'Error tidak diketahui');
    }
  } catch (err) {
    console.error('[ERROR] Gagal menyimpan data pasien ke server:', err);
    errorEl.textContent = 'Gagal menyimpan. Pastikan koneksi server backend terhubung.';
  }
}

if (socket) {
  // Ambil ulang semua bed setiap (re)connect agar update yang terlewat saat putus tetap muncul.
  socket.on('connect', fetchInitialBeds);

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
});
