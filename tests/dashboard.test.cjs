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
  bedsData.A1 = {status: 'green', vitals: {},
    patient_name: '<img src=x onerror=alert(1)>', relative_name: 'A & B'};
  displayBedDetail('A1');
`, context);
const html = elements.bedDetailContent.innerHTML;
assert.ok(html.includes('&lt;img src=x onerror=alert(1)&gt;'));
assert.ok(!html.includes('<img src=x'));
assert.ok(html.includes('A &amp; B'));
assert.equal(elements.panelBedTitle.textContent, 'DETAIL BED — A1');
console.log('Dashboard displays patient input as text, not executable HTML.');
