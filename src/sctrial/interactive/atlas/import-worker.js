import { openH5ad } from './h5-reader.js';
import { buildAtlas, listPathways } from './study-builder.js';

let reader;
let queued = Promise.resolve();
self.onmessage = ({ data }) => {
  queued = queued.then(async () => {
    const { id, type } = data;
    try {
      let result;
      if (type === 'open') {
        reader = await openH5ad(data.file);
        result = { schema: reader.schema, pathways: await listPathways() };
      } else if (type === 'levels') {
        const values = await reader.readColumn(data.name);
        result = [...new Set(values.filter((v) => v !== null && v !== ''))];
      } else if (type === 'genes') {
        result = await reader.genes(data.layer, data.symbolColumn);
      } else if (type === 'build') {
        result = await buildAtlas(reader, data.config, (message) =>
          self.postMessage({ id, progress: message })
        );
      } else {
        throw Error('Unknown atlas import operation.');
      }
      self.postMessage({ id, result }, type === 'build' ? [result.points] : []);
    } catch (error) {
      self.postMessage({ id, error: error?.message || 'The H5AD could not be read.' });
    }
  });
};
