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
