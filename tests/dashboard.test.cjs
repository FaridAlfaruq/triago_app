const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const elements = { panelBedTitle: {}, bedDetailContent: {} };
const context = vm.createContext({
  document: { getElementById: id => elements[id], addEventListener() {} },
  console: { warn() {} }, setInterval() {},
});
vm.runInContext(fs.readFileSync(path.join(__dirname, '../website/dashboard.js'), 'utf8'), context);
vm.runInContext(`
  bedsData.A1 = {status: 'green', vitals: {}, pain_score: 0,
    patient_name: '<img src=x onerror=alert(1)>', relative_name: 'A & B'};
  displayBedDetail('A1');
`, context);
const html = elements.bedDetailContent.innerHTML;
assert.ok(html.includes('&lt;img src=x onerror=alert(1)&gt;'));
assert.ok(!html.includes('<img src=x'));
assert.ok(html.includes('A &amp; B'));
assert.ok(html.includes('0 / 10'));
vm.runInContext('bedsData.A1.pain_score = null; displayBedDetail("A1");', context);
assert.ok(!elements.bedDetailContent.innerHTML.includes('null / 10'));
assert.ok(!elements.bedDetailContent.innerHTML.includes('undefined / 10'));
assert.equal(elements.panelBedTitle.textContent, 'DETAIL BED — A1');
vm.runInContext(`
  bedsData.A1.patient_details = {chief_complaint: '<script>x</script>', payer: 'bpjs', sex: 'P'};
  displayBedDetail('A1');
`, context);
const detailHtml = elements.bedDetailContent.innerHTML;
assert.ok(detailHtml.includes('&lt;script&gt;x&lt;/script&gt;'));
assert.ok(!detailHtml.includes('<script>'));
assert.ok(detailHtml.includes('BPJS Kesehatan'));
assert.ok(detailHtml.includes('Perempuan'));
console.log('Dashboard escapes patient input and distinguishes measured zero pain from unknown.');

// Denah U dibentuk dari zona: kiri Resusitasi, bawah Darurat, kanan Non-Darurat.
const layout = vm.runInContext(`
  const beds = {};
  ['red','red','red','yellow','yellow','yellow','yellow','yellow','yellow','green','green','green']
    .forEach((zone, i) => { beds[String(i + 1).padStart(2, '0')] = {zone, status: 'empty'}; });
  JSON.stringify(floorLayout(beds));
`, context);
const u = JSON.parse(layout);
assert.equal(u.cols, 6);
assert.equal(u.rows, 4);
assert.deepEqual(u.positions['01'], [1, 1]);
assert.deepEqual(u.positions['04'], [4, 1]);
assert.deepEqual(u.positions['09'], [4, 6]);
assert.deepEqual(u.positions['10'], [3, 6]);
assert.deepEqual(u.positions['12'], [1, 6]);
const big = JSON.parse(vm.runInContext(`
  const more = {};
  ['red','red','red','red','yellow','yellow','green','green','green','green','green']
    .forEach((zone, i) => { more[String(i + 1).padStart(2, '0')] = {zone, status: 'empty'}; });
  JSON.stringify(floorLayout(more));
`, context));
assert.equal(big.cols, 3);
assert.equal(big.rows, 6);
assert.deepEqual(big.positions['05'], [6, 1]);  // sisi bawah: 2 bed di tengah 3 kolom
assert.deepEqual(big.positions['11'], [1, 3]);
console.log('Dashboard builds the U floor plan from bed zones.');

// Di balik reverse proxy sub-path, API dan Socket.IO harus ikut sub-path.
const calls = { io: null, fetch: [] };
const subContext = vm.createContext({
  document: { getElementById: id => elements[id], addEventListener() {} },
  console: { warn() {}, error() {} }, setInterval() {},
  location: { pathname: '/informatics/triago/' },
  io: options => { calls.io = options; return { on() {} }; },
  fetch: url => { calls.fetch.push(url); return Promise.resolve({ ok: false }); },
});
vm.runInContext(fs.readFileSync(path.join(__dirname, '../website/dashboard.js'), 'utf8'), subContext);
assert.equal(calls.io.path, '/informatics/triago/socket.io');
vm.runInContext('fetchInitialBeds()', subContext);
assert.ok(calls.fetch.length > 0 && calls.fetch.every(url => !url.startsWith('/')));
assert.ok(!fs.readFileSync(path.join(__dirname, '../website/dashboard.js'), 'utf8').match(/fetch\(['`]\//));
console.log('Dashboard uses relative URLs that work under a sub-path.');
