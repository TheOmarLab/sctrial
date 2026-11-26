/* Local participant-level summaries. A missing specimen is represented by an omitted row. */
(() => {
  'use strict';

  const required = ['participant', 'arm', 'visit', 'population', 'cells'];
  const maxRows = 50000;
  const maxFeatures = 32;
  const maxBytes = 8 * 1024 * 1024;
  const decimal = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;

  function fieldLabel(value, name, line) {
    const result = value.trim();
    if (!result) throw new Error('Row ' + line + ' has an empty ' + name + '.');
    if (result.length > 160 || /[\u0000-\u001f\u007f]/.test(result)) {
      throw new Error(
        'Row ' + line + ' has an invalid ' + name + '. Use a single line of up to 160 characters.',
      );
    }
    return result;
  }

  function parseRows(text, accept) {
    let field = '',
      fields = [],
      quoted = false,
      closed = false;
    let line = 1,
      start = 1;

    function endField() {
      fields.push(field);
      if (fields.length > required.length + maxFeatures) {
        throw new Error(
          'Row ' + start + ' has too many fields. Include up to ' + maxFeatures + ' score columns.',
        );
      }
      field = '';
      closed = false;
    }

    function endRow() {
      endField();
      if (fields.length !== 1 || fields[0].trim()) accept(fields, start);
      fields = [];
    }

    for (let i = 0; i < text.length; i++) {
      const character = text[i];
      if (character === '\r' || character === '\n') {
        if (character === '\r' && text[i + 1] === '\n') i++;
        if (quoted) field += '\n';
        else endRow();
        line++;
        if (!quoted) start = line;
      } else if (quoted) {
        if (character !== '"') field += character;
        else if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          quoted = false;
          closed = true;
        }
      } else if (character === ',') {
        endField();
      } else if (closed) {
        if (character !== ' ' && character !== '\t') {
          throw new Error(
            'Row ' + line + ' has text after a closing quote. Use a comma between fields.',
          );
        }
      } else if (character === '"') {
        if (field.trim())
          throw new Error('Row ' + line + ' has a misplaced quote. Enclose the whole field in quotes.');
        field = '';
        quoted = true;
      } else field += character;
    }
    if (quoted) throw new Error('Row ' + start + ' has an unclosed quoted field.');
    endRow();
  }

  function readStudyCSV(text, fileName) {
    if (typeof text !== 'string') throw new Error('Choose a UTF-8 CSV file.');
    if (text.length > maxBytes || new Blob([text]).size > maxBytes) {
      throw new Error(
        'This preview supports CSV files up to 8 MiB. Select a smaller set of participant summaries.',
      );
    }
    const groups = new Set(),
      people = new Map(),
      visits = new Set(),
      populations = new Set();
    const keys = new Set(),
      observations = [];
    let positions, features, columnCount;

    function readHeader(fields, line) {
      const headers = fields.map((value) => fieldLabel(value, 'column name', line));
      const names = headers.map((name) => name.toLowerCase());
      if (new Set(names).size !== names.length)
        throw new Error('The header has duplicate column names. Give each column a unique name.');
      const missing = required.filter((name) => !names.includes(name));
      if (missing.length) throw new Error('Add the required CSV columns ' + missing.join(', ') + '.');
      positions = Object.fromEntries(required.map((name) => [name, names.indexOf(name)]));
      features = headers.flatMap((label, index) =>
        required.includes(names[index]) ? [] : [{ id: 'score_' + (index + 1), label, index }],
      );
      if (!features.length)
        throw new Error('Include at least one numeric score column after the study fields.');
      columnCount = fields.length;
    }

    function readObservation(fields, line) {
      if (observations.length >= maxRows)
        throw new Error('This preview supports up to 50,000 participant-level rows.');
      if (fields.length !== columnCount) {
        throw new Error(
          'Row ' + line + ' has ' + fields.length + ' fields. The header has ' + columnCount + '.',
        );
      }
      const id = fieldLabel(fields[positions.participant], 'participant ID', line);
      const arm = fieldLabel(fields[positions.arm], 'group label', line);
      const visit = fieldLabel(fields[positions.visit], 'visit label', line);
      const population = fieldLabel(fields[positions.population], 'population label', line);
      const previous = people.get(id);
      if (previous !== undefined && previous !== arm) {
        throw new Error(
          'Row ' + line + ' assigns ' + id + ' to a second group. Use one group per participant.',
        );
      }
      if (groups.size === 2 && !groups.has(arm)) {
        throw new Error(
          'Row ' + line + ' adds a third group. Import one or two groups for this comparison.',
        );
      }
      const key = JSON.stringify([id, visit, population]);
      if (keys.has(key)) {
        throw new Error(
          'Row ' +
            line +
            ' repeats ' +
            id +
            ' / ' +
            visit +
            ' / ' +
            population +
            '. Combine replicates before import.',
        );
      }
      const countText = fields[positions.cells].trim();
      const cells = Number(countText);
      if (!decimal.test(countText) || !Number.isSafeInteger(cells) || cells < 1) {
        throw new Error(
          'Row ' + line + ' requires a positive whole cell count. Omit rows with no observed cells.',
        );
      }
      const scores = {};
      for (const feature of features) {
        const value = fields[feature.index].trim();
        const score = Number(value);
        if (!decimal.test(value) || !Number.isFinite(score)) {
          throw new Error(
            'Row ' + line + ' requires a finite numeric value in “' + feature.label + '”.',
          );
        }
        scores[feature.id] = score;
      }
      keys.add(key);
      groups.add(arm);
      people.set(id, arm);
      visits.add(visit);
      populations.add(population);
      observations.push({ id, arm, visit, population, cells, scores });
    }

    parseRows(text.replace(/^\uFEFF/, ''), (fields, line) => {
      if (positions) readObservation(fields, line);
      else readHeader(fields, line);
    });
    if (!positions) throw new Error('The CSV is empty. Start with the example template.');
    if (!observations.length)
      throw new Error('The CSV contains a header only. Add at least one participant observation.');
    const stem = String(fileName || '')
      .split(/[\\/]/)
      .pop()
      .replace(/\.csv$/i, '')
      .replace(/[\u0000-\u001f\u007f]/g, '')
      .trim();
    return {
      title: stem.slice(0, 100) || 'Imported study',
      source: 'import',
      participants: Array.from(people, ([id, arm]) => ({ id, arm })),
      visits: Array.from(visits),
      populations: Array.from(populations),
      features: features.map(({ id, label }) => ({ id, label })),
      observations,
      results: {},
    };
  }

  function studyCSVTemplate() {
    const lines = ['participant,arm,visit,population,cells,Interferon response,Cytotoxicity'];
    const visits = ['Baseline', 'Week 4', 'Week 12'];
    const populations = ['CD8 T cells', 'B cells'];
    for (let person = 0; person < 6; person++) {
      const first = person < 3;
      const participant = (first ? 'A' : 'B') + '0' + ((person % 3) + 1);
      for (let t = 0; t < visits.length; t++) {
        if (person === 5 && t === 2) continue;
        for (let c = 0; c < populations.length; c++) {
          const baseline = (person % 3) * 0.08;
          const response = 0.62 + baseline + (first ? 0.27 : 0.13) * t - c * 0.11;
          const cytotoxicity = 0.88 + baseline + (first ? 0.16 : 0.09) * t - c * 0.49;
          lines.push(
            [
              participant,
              first ? 'Arm A' : 'Arm B',
              visits[t],
              populations[c],
              32 + person * 3 + t * 4 + c,
              response.toFixed(2),
              cytotoxicity.toFixed(2),
            ].join(','),
          );
        }
      }
    }
    return lines.join('\r\n') + '\r\n';
  }

  globalThis.readStudyCSV = readStudyCSV;
  globalThis.studyCSVTemplate = studyCSVTemplate;
})();
