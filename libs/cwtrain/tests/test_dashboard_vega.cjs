// SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
// SPDX-License-Identifier: Apache-2.0
// SPDX-PackageName: inference-cookbook

// Execute the shipped Vega dataflow, using synthetic sparse W&B history.
// Node/Vega are test-only dependencies; native panels run inside W&B.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {execFileSync} = require('node:child_process');
const vega = require(process.env.VEGA_RUNTIME || 'vega');
const panels = JSON.parse(execFileSync(process.env.PYTHON || 'python', ['-c',
  "import json; from cwtrain.dashboard.native import panels; print(json.dumps(panels('miles', None)))"
], {encoding: 'utf8'}));

function create(kind, rows) {
  const panel = panels.find(p => p.kind === kind);
  let text = fs.readFileSync(`cwtrain/dashboard/specs/${kind}.json`, 'utf8');
  for (const [key, value] of Object.entries(panel.fields)) {
    text = text.split('${field:' + key + '}').join(value);
  }
  for (const [key, value] of Object.entries(panel.strings)) {
    text = text.split('${string:' + key + '}').join(value);
  }
  return new vega.View(vega.parse(JSON.parse(text)), {renderer: 'none'})
    .insert('wandb', rows).run();
}

// Deliberately use optimizer-scaled rollout/step values. The selector must use
// the actual timeline step mapping, not silently compare two different axes.
const folded = [];
let history = 0;
function row(id, values) {
  for (const [key, val] of Object.entries(values)) {
    folded.push({id, name: id, _step: history, key, val});
  }
  history++;
}
for (const id of ['run-a', 'run-b']) {
  for (let step = 0; step < 3; step++) {
    const base = 100 + step * 20;
    row(id, {'timeline/step': step, 'rollout/step': step * 2,
      'timeline/rollout/start': base, 'timeline/rollout/end': base + 8,
      'timeline/reward/start': base + 8, 'timeline/reward/end': base + 8});
    row(id, {'rollout/step': step * 2, 'rollout/raw_reward': step / 4});
    for (let pass = 0; pass < 2; pass++) {
      row(id, {'timeline/step': step, 'rollout/step': step * 2,
        'timeline/update/start': base + 9 + pass, 'timeline/update/end': base + 10 + pass});
    }
  }
}
// NaN placeholders and absent step IDs must not become zero-valued data.
row('run-a', {'timeline/step': NaN, 'timeline/update/start': 163,
  'timeline/update/end': 164, 'rollout/raw_reward': null});
row('run-a', {'rollout/step': 4, 'rollout/raw_reward': 0.9}); // latest duplicate wins
const timeline = create('timeline', folded);
assert.equal(timeline.data('spans').length, 25);
assert.equal(timeline.data('reward_points').length, 6);
assert.deepEqual(timeline.data('reward_points').filter(d => d.id === 'run-a')
  .map(d => [d.step, d.reward]), [[0, 0], [1, 0.25], [2, 0.9]]);
assert.equal(timeline.data('unassigned_spans').length, 1);
assert.equal(timeline.data('spans').filter(d => d.end === d.start).length, 6);
assert.equal(timeline.data('spans').filter(d => d.duration_label === '0 s').length, 6);
timeline.signal('selectionMode', 'steps').signal('brush', [1, 1]).run();
assert.equal(timeline.data('selected_spans').length, 8);
assert.ok(timeline.data('selected_spans').every(d => d.step === 1));
timeline.width(600).run();
assert.equal(timeline.data('selected_spans').length, 8); // resizing retains step range
timeline.signal('selectionMode', 'missing').run();
assert.equal(timeline.data('selected_spans').length, 1);
timeline.signal('selectionMode', 'all').run();
assert.equal(timeline.data('selected_spans').length, 25);

// If two rollout IDs share a logged reward step, no unique mapping exists.
const ambiguous = [...folded,
  {id:'run-a',name:'run-a',_step:999,key:'timeline/step',val:9},
  {id:'run-a',name:'run-a',_step:999,key:'rollout/step',val:4}];
assert.equal(create('timeline', ambiguous).data('reward_points').length, 5);
assert.equal(create('timeline', []).data('spans').length, 0);

const overview = create('overview', [
  {id: 'a', name: 'First run', createdAt: '2026-01-01T00:00:00Z',
    hf_checkpoint: '/models/example', advantage_estimator: 'grpo',
    use_kl_loss: false, kl_loss_coef: 0, 'rollout/raw_reward': 0},
  {id: 'b', name: 'Second run'},
]);
assert.equal(overview.data('runs').length, 2);
assert.equal(overview.data('cells').length, 22);
const cells = overview.data('cells');
assert.equal(cells.find(d => d.id === 'a' && d.field === 'kl_value').value,
  'Disabled · coefficient 0');
assert.equal(cells.find(d => d.id === 'a' && d.field === 'reward_value').value, '0');
assert.equal(cells.find(d => d.id === 'b' && d.field === 'reward_value').value, 'Not recorded');
assert.equal(cells.find(d => d.id === 'a' && d.field === 'algorithm_value').value, 'Miles / grpo');

(async () => {
  for (const view of [timeline, overview]) {
    for (const width of [600, 1200]) {
      view.width(width).run();
      assert.ok((await view.toSVG()).startsWith('<svg'));
    }
    view.finalize();
  }
  console.log('Native Vega: sparse history, step alignment, repeated/zero spans, selection, missing values, and comparison passed.');
})().catch(error => {console.error(error); process.exitCode = 1;});
