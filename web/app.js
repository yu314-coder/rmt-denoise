// Tab switching
const tabs = document.querySelectorAll('.tab');
const contents = document.querySelectorAll('.tab-content');
tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    tabs.forEach(t => t.classList.remove('active'));
    contents.forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.tab).classList.add('active');
  });
});

function setHTML(slot, html) {
  const el = document.getElementById(slot);
  if (el) el.innerHTML = html || '';
}

// Track resize handlers to avoid duplicates
const resizeHandlers = new Map();

function plotFromFigure(target, fig) {
  const el = document.getElementById(target);
  if (!el) return;

  // Remove empty class and add has-plot class
  el.classList.remove('empty');
  el.classList.add('has-plot');
  el.innerHTML = '';

  const layout = fig.layout || {};
  layout.autosize = true;
  layout.height = 450;
  layout.margin = layout.margin || {};
  Object.assign(layout.margin, { l: 60, r: 40, t: 60, b: 50 });

  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d']
  };

  Plotly.newPlot(target, fig.data, layout, config);

  // Remove old resize handler if exists
  if (resizeHandlers.has(target)) {
    window.removeEventListener('resize', resizeHandlers.get(target));
  }

  // Create new resize handler
  const resizeHandler = () => {
    const plotDiv = document.getElementById(target);
    if (plotDiv && plotDiv.data) {
      Plotly.Plots.resize(target);
    }
  };

  resizeHandlers.set(target, resizeHandler);
  window.addEventListener('resize', resizeHandler);
}

function setProgress(barId, textId, progressList) {
  const bar = document.getElementById(barId);
  const textEl = document.getElementById(textId);
  if (!bar || !textEl) return;
  const vals = (progressList || []).map(p => {
    if (typeof p === 'object' && p !== null && ('step' in p) && ('total' in p)) {
      return { pct: Math.round((p.step / p.total) * 100), label: `${p.step}/${p.total} (${Math.round((p.step/p.total)*100)}%)` };
    }
    return { pct: 0, label: '' };
  });
  const last = vals.length ? vals[vals.length - 1] : { pct: 0, label: '' };
  bar.style.width = Math.min(100, last.pct || 0) + '%';
  textEl.innerHTML = last.label || '';
}

function apiReady() {
  return window.pywebview && window.pywebview.api;
}

function ensureApi(slotStatus) {
  if (!apiReady()) {
    setHTML(slotStatus, '<span style="color:red;">API not ready yet. Please wait a second.</span>');
    return false;
  }
  return true;
}

async function runIm() {
  setHTML('slot1', 'Running...');
  setHTML('status1', '');
  setHTML('explain1', '');
  setProgress('bar1', 'status1', []);
  if (!ensureApi('status1')) return;
  try {
    const payload = {
      beta: Number(document.getElementById('beta_z').value),
      y: Number(document.getElementById('y_z').value),
      a: Number(document.getElementById('a_z').value),
      z_min: Number(document.getElementById('zmin_z').value),
      z_max: Number(document.getElementById('zmax_z').value),
      points: Number(document.getElementById('points_z').value)
    };
    const res = await window.pywebview.api.im_vs_z(payload);

    // Plot using raw data
    const el = document.getElementById('slot1');
    el.classList.remove('empty');
    el.classList.add('has-plot');
    el.innerHTML = '';

    const {z_values, ims_values, real_values} = res.data;
    const {a, y, beta} = res.params;

    // Create traces for Im(s)
    const colors = ["#ef553b", "#2b8aef", "#1fbf68"];
    const traces = [];
    for (let i = 0; i < 3; i++) {
      traces.push({
        x: z_values,
        y: ims_values.map(row => row[i]),
        mode: 'lines',
        name: `Im s${i+1}`,
        line: {color: colors[i], width: 2}
      });
    }

    const layout = {
      title: `Im(s) vs z (beta=${beta.toFixed(3)}, y=${y.toFixed(3)}, a=${a.toFixed(3)})`,
      xaxis: {
        title: 'z',
        autorange: true,
        color: '#e2e8f0',
        gridcolor: '#1e293b'
      },
      yaxis: {
        title: 'Im(s)',
        autorange: true,
        color: '#e2e8f0',
        gridcolor: '#1e293b'
      },
      hovermode: 'x unified',
      height: 450,
      margin: {l: 60, r: 40, t: 60, b: 50},
      plot_bgcolor: '#0f172a',
      paper_bgcolor: '#0f172a',
      font: {
        color: '#e2e8f0'
      }
    };

    const config = {
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };

    Plotly.newPlot('slot1', traces, layout, config);

    setProgress('bar1', 'status1', res.progress || []);
    setHTML('explain1', res.explanation || '');
  } catch (err) {
    setHTML('slot1', '<span style="color:red;">'+err+'</span>');
  }
}

async function runBeta() {
  setHTML('slot2', 'Running...');
  setHTML('status2', '');
  setHTML('explain2', '');
  setProgress('bar2', 'status2', []);
  if (!ensureApi('status2')) return;
  try {
    const payload = {
      z: Number(document.getElementById('z_beta').value),
      y: Number(document.getElementById('y_beta').value),
      a: Number(document.getElementById('a_beta').value),
      beta_min: Number(document.getElementById('beta_min').value),
      beta_max: Number(document.getElementById('beta_max').value),
      points: Number(document.getElementById('points_beta').value)
    };
    const res = await window.pywebview.api.roots_vs_beta(payload);

    const {beta_values, ims_values, real_values, discriminants} = res.data;
    const {z, y, a} = res.params;
    const colors = ["#ef553b", "#2b8aef", "#1fbf68"];

    // Plot Im(s) vs beta
    const el2 = document.getElementById('slot2');
    el2.classList.remove('empty');
    el2.classList.add('has-plot');
    el2.innerHTML = '';

    const imTraces = [];
    for (let i = 0; i < 3; i++) {
      imTraces.push({
        x: beta_values,
        y: ims_values.map(row => row[i]),
        mode: 'lines',
        name: `Im s${i+1}`,
        line: {color: colors[i], width: 2}
      });
    }

    Plotly.newPlot('slot2', imTraces, {
      title: `Im(s) vs beta (z=${z.toFixed(3)}, y=${y.toFixed(3)}, a=${a.toFixed(3)})`,
      xaxis: {title: 'beta', autorange: true, color: '#e2e8f0', gridcolor: '#1e293b'},
      yaxis: {title: 'Im(s)', autorange: true, color: '#e2e8f0', gridcolor: '#1e293b'},
      hovermode: 'x unified',
      height: 450,
      margin: {l: 60, r: 40, t: 60, b: 50},
      plot_bgcolor: '#0f172a',
      paper_bgcolor: '#0f172a',
      font: { color: '#e2e8f0' }
    }, {responsive: true, displayModeBar: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d']});

    // Plot Re(s) vs beta
    const el3 = document.getElementById('slot3');
    el3.classList.remove('empty');
    el3.classList.add('has-plot');
    el3.innerHTML = '';

    const reTraces = [];
    for (let i = 0; i < 3; i++) {
      reTraces.push({
        x: beta_values,
        y: real_values.map(row => row[i]),
        mode: 'lines',
        name: `Re s${i+1}`,
        line: {color: colors[i], width: 2}
      });
    }

    Plotly.newPlot('slot3', reTraces, {
      title: `Re(s) vs beta (z=${z.toFixed(3)}, y=${y.toFixed(3)}, a=${a.toFixed(3)})`,
      xaxis: {title: 'beta', autorange: true, color: '#e2e8f0', gridcolor: '#1e293b'},
      yaxis: {title: 'Re(s)', autorange: true, color: '#e2e8f0', gridcolor: '#1e293b'},
      hovermode: 'x unified',
      height: 450,
      margin: {l: 60, r: 40, t: 60, b: 50},
      plot_bgcolor: '#0f172a',
      paper_bgcolor: '#0f172a',
      font: { color: '#e2e8f0' }
    }, {responsive: true, displayModeBar: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d']});

    // Plot discriminant
    const el4 = document.getElementById('slot4');
    el4.classList.remove('empty');
    el4.classList.add('has-plot');
    el4.innerHTML = '';

    Plotly.newPlot('slot4', [{
      x: beta_values,
      y: discriminants,
      mode: 'lines',
      name: 'Cubic Discriminant',
      line: {color: 'white', width: 2}
    }], {
      title: `Discriminant vs beta (z=${z.toFixed(3)}, y=${y.toFixed(3)}, a=${a.toFixed(3)})`,
      xaxis: {title: 'beta', autorange: true, color: '#e2e8f0', gridcolor: '#1e293b'},
      yaxis: {title: 'Discriminant', autorange: true, color: '#e2e8f0', gridcolor: '#1e293b'},
      hovermode: 'x unified',
      height: 450,
      margin: {l: 60, r: 40, t: 60, b: 50},
      plot_bgcolor: '#0f172a',
      paper_bgcolor: '#0f172a',
      font: { color: '#e2e8f0' }
    }, {responsive: true, displayModeBar: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d']});

    setProgress('bar2', 'status2', res.progress || []);
    setHTML('explain2', res.explanation || '');
  } catch (err) {
    setHTML('slot2', '<span style="color:red;">'+err+'</span>');
  }
}

async function runEigen() {
  setHTML('slot6', 'Running...');
  setHTML('status4', '');
  setHTML('explain4', '');
  setProgress('bar4', 'status4', []);
  if (!ensureApi('status4')) return;
  try {
    const payload = {
      beta: Number(document.getElementById('beta_eig').value),
      a: Number(document.getElementById('a_eig').value),
      n: Number(document.getElementById('n_eig').value),
      p: Number(document.getElementById('p_eig').value),
      seed: Number(document.getElementById('seed_eig').value)
    };
    const res = await window.pywebview.api.eigen_distribution(payload);

    const {eigenvalues, kde_x, kde_y} = res.data;

    // Plot eigenvalue distribution
    const el6 = document.getElementById('slot6');
    el6.classList.remove('empty');
    el6.classList.add('has-plot');
    el6.innerHTML = '';

    const traces = [
      {
        x: eigenvalues,
        type: 'histogram',
        histnorm: 'probability density',
        name: 'Histogram',
        marker: {
          color: 'rgba(59, 130, 246, 0.7)',
          line: {
            color: 'rgba(59, 130, 246, 1)',
            width: 1
          }
        }
      },
      {
        x: kde_x,
        y: kde_y,
        mode: 'lines',
        name: 'KDE',
        line: {color: '#60a5fa', width: 2}
      }
    ];

    Plotly.newPlot('slot6', traces, {
      title: `Eigenvalue distribution (y=${payload.p / payload.n}, beta=${payload.beta}, a=${payload.a})`,
      xaxis: {title: 'Eigenvalue', autorange: true, color: '#e2e8f0', gridcolor: '#1e293b'},
      yaxis: {title: 'Density', autorange: true, color: '#e2e8f0', gridcolor: '#1e293b'},
      hovermode: 'closest',
      height: 450,
      margin: {l: 60, r: 40, t: 60, b: 50},
      plot_bgcolor: '#0f172a',
      paper_bgcolor: '#0f172a',
      font: { color: '#e2e8f0' },
      legend: {
        font: { color: '#e2e8f0' }
      }
    }, {responsive: true, displayModeBar: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d']});

    document.getElementById('eig_stats').innerHTML = `
      <div class="metric">min: ${res.stats.min.toFixed(4)}</div>
      <div class="metric">max: ${res.stats.max.toFixed(4)}</div>
      <div class="metric">mean: ${res.stats.mean.toFixed(4)}</div>
      <div class="metric">std: ${res.stats.std.toFixed(4)}</div>
    `;
    setProgress('bar4', 'status4', res.progress || []);
    setHTML('explain4', res.explanation || '');
  } catch (err) {
    setHTML('slot6', '<span style="color:red;">'+err+'</span>');
  }
}

// Image Download & Management functions
let currentImageIndex = 0;
let totalImages = 0;
let processedResults = [];

document.getElementById('local_folder').addEventListener('change', function(e) {
  const files = e.target.files;
  if (files.length > 0) {
    // Get folder name from first file's path
    const path = files[0].webkitRelativePath || files[0].name;
    const folderName = path.split('/')[0];
    const imageFiles = Array.from(files).filter(f => /\.(png|jpg|jpeg|gif|bmp)$/i.test(f.name));
    document.getElementById('folder_count').innerHTML = `Folder: ${folderName} (${imageFiles.length} images)`;
  } else {
    document.getElementById('folder_count').innerHTML = 'No folder selected';
  }
});

document.getElementById('local_files').addEventListener('change', function(e) {
  const files = e.target.files;
  if (files.length > 0) {
    const imageFiles = Array.from(files).filter(f => /\.(png|jpg|jpeg|gif|bmp)$/i.test(f.name));
    const totalSize = imageFiles.reduce((sum, f) => sum + f.size, 0);
    const sizeMB = (totalSize / (1024 * 1024)).toFixed(2);
    document.getElementById('files_count').innerHTML = `${imageFiles.length} image(s) selected (${sizeMB} MB)`;
  } else {
    document.getElementById('files_count').innerHTML = 'No files selected';
  }
});

function toggleMethodParams() {
  const method1 = document.getElementById('method1').value;
  const method2 = document.getElementById('method2').value;
  const method1Params = document.getElementById('method1_params');
  const method2Params = document.getElementById('method2_params');

  // Show manual parameters only for gen_manual method
  if (method1Params) {
    method1Params.style.display = method1 === 'gen_manual' ? 'block' : 'none';
  }
  if (method2Params) {
    method2Params.style.display = method2 === 'gen_manual' ? 'block' : 'none';
  }
}

function toggleNoiseParams() {
  const noiseType = document.getElementById('noise_type').value;
  document.getElementById('laplacian_params').style.display = noiseType === 'laplacian' ? 'block' : 'none';
  document.getElementById('mog_params').style.display = noiseType === 'mixture_gaussian' ? 'block' : 'none';
}

async function importLocalFolder() {
  if (!ensureApi('status_upload_folder')) return;
  const files = document.getElementById('local_folder').files;
  if (files.length === 0) {
    setHTML('status_upload_folder', '<span style="color:orange;">No folder selected</span>');
    return;
  }

  try {
    // Get folder name from first file's path
    const path = files[0].webkitRelativePath || files[0].name;
    const sourceFolderName = path.split('/')[0];

    // Filter image files only
    const imageFiles = Array.from(files).filter(f => /\.(png|jpg|jpeg|gif|bmp)$/i.test(f.name));

    if (imageFiles.length === 0) {
      setHTML('status_upload_folder', '<span style="color:orange;">No image files found in folder</span>');
      return;
    }

    const fileData = [];
    for (let i = 0; i < imageFiles.length; i++) {
      const file = imageFiles[i];
      const reader = new FileReader();
      const base64 = await new Promise((resolve) => {
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.readAsDataURL(file);
      });

      // Get relative path within the selected folder
      const fullPath = file.webkitRelativePath || file.name;
      const pathParts = fullPath.split('/');
      // Remove the first part (folder name) to get relative path
      const relativePath = pathParts.slice(1).join('/');

      fileData.push({
        name: file.name,
        path: relativePath,  // Preserve relative path
        data: base64
      });

      // Update progress bar
      const progress = Math.round(((i + 1) / imageFiles.length) * 100);
      document.getElementById('bar_upload_folder').style.width = progress + '%';
      setHTML('status_upload_folder', `Reading files: ${i + 1}/${imageFiles.length} (${progress}%)`);
    }

    const payload = {
      files: fileData,
      folder: sourceFolderName  // Use original folder name
    };

    setHTML('status_upload_folder', `Uploading ${imageFiles.length} images...`);
    const res = await window.pywebview.api.import_local_folder(payload);

    if (res && res.success) {
      setHTML('status_upload_folder', `<span style="color:green;">✔ Imported ${res.count} images to folder "${res.folder}"</span>`);
      document.getElementById('folder_count').innerHTML = 'No folder selected';
      document.getElementById('local_folder').value = '';
      document.getElementById('bar_upload_folder').style.width = '100%';
      refreshFolders();
    } else {
      const errorMsg = (res && res.error) ? res.error : 'Unknown error';
      setHTML('status_upload_folder', `<span style="color:red;">Error: ${errorMsg}</span>`);
      document.getElementById('bar_upload_folder').style.width = '0%';
    }
  } catch (err) {
    console.error('Upload error:', err);
    setHTML('status_upload_folder', '<span style="color:red;">Error: '+err.message+'</span>');
    document.getElementById('bar_upload_folder').style.width = '0%';
  }
}

async function uploadFilesToFolder() {
  if (!ensureApi('status_upload_files')) return;
  const files = document.getElementById('local_files').files;
  const destFolder = document.getElementById('upload_dest_folder').value;

  if (!destFolder || destFolder === 'Loading folders...' || destFolder === 'No folders found') {
    setHTML('status_upload_files', '<span style="color:orange;">Please select a destination folder</span>');
    return;
  }

  if (files.length === 0) {
    setHTML('status_upload_files', '<span style="color:orange;">No files selected</span>');
    return;
  }

  try {
    // Filter image files only
    const imageFiles = Array.from(files).filter(f => /\.(png|jpg|jpeg|gif|bmp)$/i.test(f.name));

    if (imageFiles.length === 0) {
      setHTML('status_upload_files', '<span style="color:orange;">No image files selected</span>');
      return;
    }

    const fileData = [];
    for (let i = 0; i < imageFiles.length; i++) {
      const file = imageFiles[i];
      const reader = new FileReader();
      const base64 = await new Promise((resolve) => {
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.readAsDataURL(file);
      });

      fileData.push({
        name: file.name,
        data: base64
      });

      // Update progress bar
      const progress = Math.round(((i + 1) / imageFiles.length) * 100);
      document.getElementById('bar_upload_files').style.width = progress + '%';
      setHTML('status_upload_files', `Reading files: ${i + 1}/${imageFiles.length} (${progress}%)`);
    }

    const payload = {
      files: fileData,
      folder: destFolder
    };

    setHTML('status_upload_files', `Uploading ${imageFiles.length} images to "${destFolder}"...`);
    const res = await window.pywebview.api.upload_files_to_folder(payload);

    if (res && res.success) {
      setHTML('status_upload_files', `<span style="color:green;">✔ Uploaded ${res.count} files to "${res.folder}"</span>`);
      document.getElementById('files_count').innerHTML = 'No files selected';
      document.getElementById('local_files').value = '';
      document.getElementById('bar_upload_files').style.width = '100%';
      refreshFolders();

      // Clear status after 3 seconds
      setTimeout(() => {
        setHTML('status_upload_files', '');
        document.getElementById('bar_upload_files').style.width = '0%';
      }, 3000);
    } else {
      const errorMsg = (res && res.error) ? res.error : 'Unknown error';
      setHTML('status_upload_files', `<span style="color:red;">Error: ${errorMsg}</span>`);
      document.getElementById('bar_upload_files').style.width = '0%';
    }
  } catch (err) {
    console.error('Upload error:', err);
    setHTML('status_upload_files', '<span style="color:red;">Error: '+err.message+'</span>');
    document.getElementById('bar_upload_files').style.width = '0%';
  }
}

async function refreshFolders() {
  // Wait for API to be ready
  if (!apiReady()) {
    console.log('API not ready, retrying in 500ms...');
    setTimeout(refreshFolders, 500);
    return;
  }

  try {
    const res = await window.pywebview.api.list_random_matrix_folders();
    const folderList = document.getElementById('folder_list');
    const processFolderList = document.getElementById('process_folder');
    const uploadDestFolder = document.getElementById('upload_dest_folder');

    if (!folderList || !processFolderList) {
      console.log('DOM elements not ready, retrying...');
      setTimeout(refreshFolders, 500);
      return;
    }

    folderList.innerHTML = '';
    processFolderList.innerHTML = '';
    if (uploadDestFolder) uploadDestFolder.innerHTML = '';

    if (res.folders.length === 0) {
      folderList.innerHTML = '<option>No folders found</option>';
      processFolderList.innerHTML = '<option>No folders found</option>';
      if (uploadDestFolder) uploadDestFolder.innerHTML = '<option>No folders found</option>';
    } else {
      res.folders.forEach(folder => {
        const opt1 = document.createElement('option');
        opt1.value = folder;
        opt1.textContent = folder;
        folderList.appendChild(opt1);

        const opt2 = document.createElement('option');
        opt2.value = folder;
        opt2.textContent = folder;
        processFolderList.appendChild(opt2);

        if (uploadDestFolder) {
          const opt3 = document.createElement('option');
          opt3.value = folder;
          opt3.textContent = folder;
          uploadDestFolder.appendChild(opt3);
        }
      });
    }
    console.log(`Loaded ${res.folders.length} folders`);
  } catch (err) {
    console.error('Error refreshing folders:', err);
    // Retry on error
    setTimeout(refreshFolders, 1000);
  }
}

async function viewFolder() {
  if (!ensureApi('status_browse')) return;
  const folderList = document.getElementById('folder_list');
  const selectedFolder = folderList.value;

  if (!selectedFolder || selectedFolder === 'Loading folders...' || selectedFolder === 'No folders found') {
    return;
  }

  try {
    const res = await window.pywebview.api.get_folder_contents(selectedFolder);
    const contentsDiv = document.getElementById('folder_contents');
    const folderInfo = document.getElementById('folder_info');

    // Update folder info
    const totalFiles = res.files.length;
    const displayCount = Math.min(totalFiles, 50);
    if (totalFiles > 50) {
      folderInfo.innerHTML = `📁 ${selectedFolder} - Showing first ${displayCount} of ${totalFiles} images`;
    } else if (totalFiles > 0) {
      folderInfo.innerHTML = `📁 ${selectedFolder} - ${totalFiles} image(s)`;
    } else {
      folderInfo.innerHTML = `📁 ${selectedFolder}`;
    }

    if (res.files.length === 0) {
      contentsDiv.innerHTML = '<p style="text-align: center; color: #94a3b8; padding: 40px;"><span style="font-size: 32px; display: block; margin-bottom: 8px;">📂</span>Folder is empty</p>';
    } else {
      let html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 12px;">';
      res.files.forEach((file, idx) => {
        html += `
          <div style="text-align: center; padding: 8px; background: white; border-radius: 6px; border: 1px solid #e2e8f0;">
            <img src="data:image/png;base64,${file.data}" style="width: 100%; height: 120px; object-fit: contain; border-radius: 4px; background: #f8fafc;" />
            <div style="font-size: 11px; color: #64748b; margin-top: 6px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${file.name}">${file.name}</div>
          </div>
        `;
      });
      html += '</div>';
      contentsDiv.innerHTML = html;
    }
  } catch (err) {
    document.getElementById('folder_contents').innerHTML = '<p style="color: red; text-align: center; padding: 40px;">Error loading folder contents</p>';
    document.getElementById('folder_info').innerHTML = '';
  }
}

async function createNewFolder() {
  if (!ensureApi('status_create')) return;
  const folderName = document.getElementById('new_folder_name').value.trim();

  if (!folderName) {
    setHTML('status_create', '<span style="color:orange;">Please enter a folder name</span>');
    return;
  }

  try {
    setHTML('status_create', 'Creating folder...');
    document.getElementById('bar_create').style.width = '50%';

    const res = await window.pywebview.api.create_folder({ folder_name: folderName });

    if (res.success) {
      setHTML('status_create', `<span style="color:green;">✔ Folder "${res.folder}" created successfully</span>`);
      document.getElementById('bar_create').style.width = '100%';
      document.getElementById('new_folder_name').value = '';
      refreshFolders();

      // Clear status after 3 seconds
      setTimeout(() => {
        setHTML('status_create', '');
        document.getElementById('bar_create').style.width = '0%';
      }, 3000);
    } else {
      setHTML('status_create', `<span style="color:red;">Error: ${res.error}</span>`);
      document.getElementById('bar_create').style.width = '0%';
    }
  } catch (err) {
    setHTML('status_create', '<span style="color:red;">'+err+'</span>');
    document.getElementById('bar_create').style.width = '0%';
  }
}

async function deleteFolder() {
  if (!ensureApi('status_browse')) return;
  const folderList = document.getElementById('folder_list');
  const selectedFolder = folderList.value;

  if (!selectedFolder || selectedFolder === 'Loading folders...' || selectedFolder === 'No folders found') {
    setHTML('status_browse', '<span style="color:orange;">Please select a folder to delete</span>');
    return;
  }

  // Confirm deletion
  const confirmed = confirm(`Are you sure you want to delete the folder "${selectedFolder}" and all its contents? This action cannot be undone.`);
  if (!confirmed) {
    return;
  }

  try {
    setHTML('status_browse', 'Deleting folder...');

    const res = await window.pywebview.api.delete_folder({ folder_name: selectedFolder });

    if (res.success) {
      setHTML('status_browse', `<span style="color:green;">✔ Folder "${res.folder}" deleted successfully</span>`);
      document.getElementById('folder_contents').innerHTML = '<p style="text-align: center; color: #94a3b8; padding: 40px;">Select a folder to view its contents</p>';
      refreshFolders();

      // Clear status after 3 seconds
      setTimeout(() => {
        setHTML('status_browse', '');
      }, 3000);
    } else {
      setHTML('status_browse', `<span style="color:red;">Error: ${res.error}</span>`);
    }
  } catch (err) {
    setHTML('status_browse', '<span style="color:red;">'+err+'</span>');
  }
}

async function processImages() {
  if (!ensureApi('status_process')) return;
  const folder = document.getElementById('process_folder').value;

  if (!folder || folder === 'Loading folders...' || folder === 'No folders found') {
    setHTML('status_process', '<span style="color:orange;">Please select a folder</span>');
    return;
  }

  try {
    // Reset info panels
    setHTML('processing_info', 'Starting processing...');
    setHTML('method1_info', 'Method 1 processing...');
    setHTML('method2_info', 'Method 2 processing...');

    const method1 = document.getElementById('method1').value;
    const method2 = document.getElementById('method2').value;

    const noiseType = document.getElementById('noise_type').value;
    const randomSeedInput = document.getElementById('random_seed').value;
    const numImagesInput = document.getElementById('num_images');
    const numImages = numImagesInput ? Number(numImagesInput.value) : 0;
    const payload = {
      folder: folder,
      method1: method1,
      method2: method2,
      noise_type: noiseType,
      laplacian_scale: Number(document.getElementById('laplacian_scale').value),
      random_seed: randomSeedInput ? Number(randomSeedInput) : null,
      num_images: numImages
    };

    // Add Mixture of Gaussians parameters if selected
    if (noiseType === 'mixture_gaussian') {
      payload.mog_weights = [
        Number(document.getElementById('mog_weight1').value),
        Number(document.getElementById('mog_weight2').value),
        Number(document.getElementById('mog_weight3').value)
      ];
      payload.mog_means = [
        Number(document.getElementById('mog_mean1').value),
        Number(document.getElementById('mog_mean2').value),
        Number(document.getElementById('mog_mean3').value)
      ];
      payload.mog_sigmas = [
        Number(document.getElementById('mog_sigma1').value),
        Number(document.getElementById('mog_sigma2').value),
        Number(document.getElementById('mog_sigma3').value)
      ];
    }

    // Add manual params for Method 1 if needed
    if (method1 === 'gen_manual') {
      payload.method1_sigma2 = Number(document.getElementById('method1_sigma2').value);
      payload.method1_a = Number(document.getElementById('method1_a').value);
      payload.method1_beta = Number(document.getElementById('method1_beta').value);
    }

    // Add manual params for Method 2 if needed
    if (method2 === 'gen_manual') {
      payload.method2_sigma2 = Number(document.getElementById('method2_sigma2').value);
      payload.method2_a = Number(document.getElementById('method2_a').value);
      payload.method2_beta = Number(document.getElementById('method2_beta').value);
    }

    setHTML('status_process', 'Processing images...');
    const res = await window.pywebview.api.process_images(payload);

    if (res.success) {
      totalImages = res.total_images;
      currentImageIndex = 0;
      processedResults = res.results;  // Store results
      displayCurrentImage(processedResults[0]);

      // Display processing details (already updated in real-time by Python)
      setHTML('status_process', `<span style="color:green;">✔ Completed! Processed ${res.total_images} images</span>`);
      updateImageCounter();
    } else {
      setHTML('status_process', `<span style="color:red;">Error: ${res.error}</span>`);
    }
  } catch (err) {
    setHTML('status_process', '<span style="color:red;">'+err+'</span>');
  }
}

function displayCurrentImage(data) {
  // Top Left: Method 1
  document.getElementById('result1').innerHTML = `<img src="data:image/png;base64,${data.method1}" style="width: 100%; height: 100%; object-fit: contain;" />`;

  // Top Right: Method 2
  document.getElementById('result2').innerHTML = `<img src="data:image/png;base64,${data.method2}" style="width: 100%; height: 100%; object-fit: contain;" />`;

  // Bottom Left: Original
  document.getElementById('result_original').innerHTML = `<img src="data:image/png;base64,${data.original}" style="width: 100%; height: 100%; object-fit: contain;" />`;

  // Bottom Right: With Noise
  document.getElementById('result_blur').innerHTML = `<img src="data:image/png;base64,${data.blurred}" style="width: 100%; height: 100%; object-fit: contain;" />`;

  // Update labels with method names and eigenvector count if available
  if (data.method1_name) {
    const eigvec1 = data.method1_eigvec !== undefined ? ` (${data.method1_eigvec} eigvec)` : '';
    document.getElementById('result1_label').textContent = data.method1_name + eigvec1;
  }
  if (data.method2_name) {
    const eigvec2 = data.method2_eigvec !== undefined ? ` (${data.method2_eigvec} eigvec)` : '';
    document.getElementById('result2_label').textContent = data.method2_name + eigvec2;
  }
  if (data.noise_type_label) {
    document.getElementById('result_blur_label').textContent = data.noise_type_label;
  }

  // Update method comparison info
  const comparisonDiv = document.getElementById('method_comparison');
  if (comparisonDiv && data.method1_name && data.method2_name) {
    comparisonDiv.innerHTML = `
      <div style="text-align: left; font-size: 13px; color: #475569;">
        <strong>Method 1:</strong> ${data.method1_name}
        <strong style="color: #2563eb;">(${data.method1_eigvec || 0} eigenvectors)</strong><br>
        <strong>Method 2:</strong> ${data.method2_name}
        <strong style="color: #2563eb;">(${data.method2_eigvec || 0} eigenvectors)</strong><br>
        <strong>Noise:</strong> ${data.noise_type_label}<br><br>
        <div style="margin-top: 12px; padding: 12px; background: #f0f9ff; border-radius: 6px;">
          Click on any image to view its eigenvalue distribution below
        </div>
      </div>
    `;
  }
}

async function prevImage() {
  if (currentImageIndex > 0) {
    currentImageIndex--;
    await loadImage(currentImageIndex);
    updateImageCounter();
  }
}

async function nextImage() {
  if (currentImageIndex < totalImages - 1) {
    currentImageIndex++;
    await loadImage(currentImageIndex);
    updateImageCounter();
  }
}

async function loadImage(index) {
  // Use cached results instead of calling backend
  if (processedResults && processedResults[index]) {
    displayCurrentImage(processedResults[index]);
  } else {
    console.error('No processed results available');
  }
}

function updateImageCounter() {
  const counter = String(currentImageIndex).padStart(3, '0');
  document.getElementById('image_counter').textContent = counter;
}

async function showImageEigenvalues(imageId) {
  if (!ensureApi('eigenvalue_title')) return;

  const imageDiv = document.getElementById(imageId);
  if (!imageDiv) return;

  // Get the image element inside the div
  const imgElement = imageDiv.querySelector('img');
  if (!imgElement) {
    setHTML('eigenvalue_title', '<span style="color: #dc2626;">No image available. Please process images first.</span>');
    document.getElementById('eigenvalue_section').style.display = 'none';
    return;
  }

  try {
    // Show the eigenvalue section
    document.getElementById('eigenvalue_section').style.display = 'block';

    // Highlight the clicked image temporarily
    imageDiv.style.boxShadow = '0 0 0 3px #3b82f6';
    setTimeout(() => { imageDiv.style.boxShadow = ''; }, 500);

    // Get image label
    const labelMap = {
      'result_original': 'Original Image',
      'result_denoised': document.getElementById('result_denoised_label')?.textContent || 'Denoised Result',
      'result_noisy': document.getElementById('result_noisy_label')?.textContent || 'With Noise'
    };
    const label = labelMap[imageId] || 'Selected Image';

    setHTML('eigenvalue_title', `Computing eigenvalue distribution for: <strong>${label}</strong>...`);
    setHTML('eigenvalue_plot', '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #64748b;">Computing eigenvalues...</div>');
    setHTML('eigenvalue_stats', '');

    // Extract base64 image data from src
    const imgSrc = imgElement.src;
    if (!imgSrc || !imgSrc.includes('base64,')) {
      setHTML('eigenvalue_title', '<span style="color: #dc2626;">Invalid image data</span>');
      return;
    }

    const base64Data = imgSrc.split('base64,')[1];

    // Call backend to compute eigenvalues
    const res = await window.pywebview.api.compute_image_eigenvalues({ image_data: base64Data });

    if (res.success) {
      setHTML('eigenvalue_title', `Eigenvalue Distribution for: <strong>${label}</strong>`);

      // Plot eigenvalue distribution with enhanced styling
      const trace1 = {
        x: res.eigenvalues,
        type: 'histogram',
        name: 'Histogram',
        marker: {
          color: '#60a5fa',
          opacity: 0.65,
          line: {
            color: '#3b82f6',
            width: 1
          }
        },
        nbinsx: 40,
        hovertemplate: 'Eigenvalue: %{x:.4f}<br>Count: %{y}<extra></extra>'
      };

      const trace2 = {
        x: res.kde_x,
        y: res.kde_y,
        type: 'scatter',
        mode: 'lines',
        name: 'Density (KDE)',
        line: {
          color: '#f43f5e',
          width: 3,
          shape: 'spline'
        },
        fill: 'tozeroy',
        fillcolor: 'rgba(244, 63, 94, 0.1)',
        hovertemplate: 'Eigenvalue: %{x:.4f}<br>Density: %{y:.4f}<extra></extra>'
      };

      const layout = {
        title: '',
        xaxis: {
          title: {
            text: 'Eigenvalue',
            font: { size: 14, weight: 600, color: '#1e293b' }
          },
          gridcolor: '#e2e8f0',
          zeroline: false
        },
        yaxis: {
          title: {
            text: 'Density / Frequency',
            font: { size: 14, weight: 600, color: '#1e293b' }
          },
          gridcolor: '#e2e8f0',
          zeroline: false
        },
        showlegend: true,
        legend: {
          x: 0.02,
          y: 0.98,
          bgcolor: 'rgba(255, 255, 255, 0.9)',
          bordercolor: '#cbd5e1',
          borderwidth: 1,
          font: { size: 12 }
        },
        autosize: true,
        height: 450,
        margin: { l: 70, r: 30, t: 20, b: 60 },
        plot_bgcolor: '#fafafa',
        paper_bgcolor: 'white',
        hovermode: 'closest'
      };

      const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d', 'toImage'],
        toImageButtonOptions: {
          format: 'png',
          filename: 'eigenvalue_distribution',
          height: 600,
          width: 1000
        }
      };

      Plotly.newPlot('eigenvalue_plot', [trace1, trace2], layout, config);

      // Display enhanced statistics
      const stats = res.stats;
      const statsHTML = `
        <div style="margin-bottom: 12px; font-weight: 600; color: #92400e; font-size: 14px;">📊 Statistical Summary</div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px;">
          <div style="background: white; padding: 10px; border-radius: 6px; border: 1px solid #fde68a;">
            <div style="font-size: 11px; color: #92400e; margin-bottom: 2px;">Count</div>
            <div style="font-weight: 600; font-size: 15px; color: #78350f;">${stats.count}</div>
          </div>
          <div style="background: white; padding: 10px; border-radius: 6px; border: 1px solid #fde68a;">
            <div style="font-size: 11px; color: #92400e; margin-bottom: 2px;">Minimum</div>
            <div style="font-weight: 600; font-size: 15px; color: #78350f;">${stats.min.toFixed(6)}</div>
          </div>
          <div style="background: white; padding: 10px; border-radius: 6px; border: 1px solid #fde68a;">
            <div style="font-size: 11px; color: #92400e; margin-bottom: 2px;">Maximum</div>
            <div style="font-weight: 600; font-size: 15px; color: #78350f;">${stats.max.toFixed(6)}</div>
          </div>
          <div style="background: white; padding: 10px; border-radius: 6px; border: 1px solid #fde68a;">
            <div style="font-size: 11px; color: #92400e; margin-bottom: 2px;">Mean (μ)</div>
            <div style="font-weight: 600; font-size: 15px; color: #78350f;">${stats.mean.toFixed(6)}</div>
          </div>
          <div style="background: white; padding: 10px; border-radius: 6px; border: 1px solid #fde68a;">
            <div style="font-size: 11px; color: #92400e; margin-bottom: 2px;">Std Dev (σ)</div>
            <div style="font-weight: 600; font-size: 15px; color: #78350f;">${stats.std.toFixed(6)}</div>
          </div>
          <div style="background: white; padding: 10px; border-radius: 6px; border: 1px solid #fde68a;">
            <div style="font-size: 11px; color: #92400e; margin-bottom: 2px;">Image Size</div>
            <div style="font-weight: 600; font-size: 15px; color: #78350f;">${res.image_size.height}×${res.image_size.width}</div>
          </div>
        </div>
      `;
      setHTML('eigenvalue_stats', statsHTML);

      // Scroll to eigenvalue section
      document.getElementById('eigenvalue_section').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } else {
      setHTML('eigenvalue_title', `<span style="color: #dc2626;">Error: ${res.error}</span>`);
      setHTML('eigenvalue_plot', '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #dc2626;">Failed to compute eigenvalues</div>');
    }
  } catch (err) {
    console.error('Error computing eigenvalues:', err);
    setHTML('eigenvalue_title', `<span style="color: #dc2626;">Error: ${err.message}</span>`);
    setHTML('eigenvalue_plot', '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #dc2626;">Error occurred</div>');
  }
}

// ========== Temp File Management Functions ==========

async function refreshTempFiles() {
  if (!ensureApi('temp_files_status')) return;

  try {
    setHTML('temp_files_status', '<span style="color:#3b82f6;">Loading temp folders...</span>');
    const res = await window.pywebview.api.list_temp_folders();

    if (res.error) {
      setHTML('temp_files_status', `<span style="color:red;">Error: ${res.error}</span>`);
      return;
    }

    if (res.folders.length === 0) {
      setHTML('temp_files_status', '<span style="color:#94a3b8;">No output folders found</span>');
      setHTML('temp_files_list', '<div style="text-align: center; color: #94a3b8; padding: 40px;">No folders in temp directory</div>');
    } else {
      setHTML('temp_files_status', `<span style="color:green;">✔ Found ${res.folders.length} folder(s)</span>`);
      renderTempFoldersList(res.folders);
    }
  } catch (err) {
    setHTML('temp_files_status', `<span style="color:red;">Error: ${err}</span>`);
  }
}

function renderTempFoldersList(folders) {
  let html = '<div style="display: flex; flex-direction: column; gap: 12px;">';

  for (const folder of folders) {
    const sizeDisplay = folder.size_mb > 0 ? `${folder.size_mb} MB` : `${(folder.size_bytes / 1024).toFixed(2)} KB`;
    // Properly escape folder name for use in onclick attributes
    const escapedName = folder.name.replace(/\\/g, '\\\\').replace(/'/g, "\\'");

    html += `
      <div style="border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px; background: white;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 12px;">
          <div style="flex: 1; min-width: 0;">
            <div style="font-weight: 600; font-size: 14px; color: #0f172a; margin-bottom: 4px; word-break: break-all;">
              📁 ${folder.name}
            </div>
            <div style="font-size: 12px; color: #64748b; display: flex; flex-wrap: wrap; gap: 12px;">
              <span>📅 ${folder.modified_str}</span>
              <span>📊 ${sizeDisplay}</span>
              <span>🖼️ ${folder.file_count} file(s)</span>
            </div>
          </div>
          <div style="display: flex; gap: 8px;">
            <button
              onclick="openTempFolder('${escapedName}')"
              style="background: #3b82f6; color: white; border: none; border-radius: 6px; padding: 8px 16px; cursor: pointer; font-size: 13px; white-space: nowrap;"
              title="Open folder in file explorer"
            >
              📂 Open
            </button>
            <button
              onclick="deleteTempFolder('${escapedName}')"
              style="background: #dc2626; color: white; border: none; border-radius: 6px; padding: 8px 16px; cursor: pointer; font-size: 13px; white-space: nowrap;"
              title="Delete this folder"
            >
              🗑️ Delete
            </button>
          </div>
        </div>
      </div>
    `;
  }

  html += '</div>';
  setHTML('temp_files_list', html);
}

async function deleteTempFolder(folderName) {
  if (!ensureApi('temp_files_status')) return;

  if (!confirm(`Delete folder "${folderName}"?\n\nThis will permanently delete all files in this folder.`)) {
    return;
  }

  try {
    setHTML('temp_files_status', `<span style="color:#3b82f6;">Deleting ${folderName}...</span>`);
    const res = await window.pywebview.api.delete_temp_folder(folderName);

    if (res.success) {
      setHTML('temp_files_status', `<span style="color:green;">✔ ${res.message}</span>`);
      // Refresh the list
      await refreshTempFiles();
    } else {
      setHTML('temp_files_status', `<span style="color:red;">Error: ${res.error}</span>`);
    }
  } catch (err) {
    setHTML('temp_files_status', `<span style="color:red;">Error: ${err}</span>`);
  }
}

async function deleteAllTempFiles() {
  if (!ensureApi('temp_files_status')) return;

  if (!confirm('Delete ALL temp folders?\n\nThis will permanently delete all output folders and files in the temp directory.\n\nThis action cannot be undone!')) {
    return;
  }

  try {
    setHTML('temp_files_status', '<span style="color:#3b82f6;">Deleting all folders...</span>');
    const res = await window.pywebview.api.delete_all_temp_folders();

    if (res.success) {
      setHTML('temp_files_status', `<span style="color:green;">✔ ${res.message}</span>`);
      // Refresh the list
      await refreshTempFiles();
    } else {
      setHTML('temp_files_status', `<span style="color:red;">Error: ${res.error}</span>`);
    }
  } catch (err) {
    setHTML('temp_files_status', `<span style="color:red;">Error: ${err}</span>`);
  }
}

async function openTempFolder(folderName) {
  if (!ensureApi('temp_files_status')) return;

  try {
    setHTML('temp_files_status', `<span style="color:#3b82f6;">Opening folder...</span>`);
    const res = await window.pywebview.api.open_temp_folder(folderName);

    if (res.success) {
      setHTML('temp_files_status', `<span style="color:green;">✔ Opened folder: ${folderName}</span>`);
    } else {
      setHTML('temp_files_status', `<span style="color:red;">Error: ${res.error}</span>`);
    }
  } catch (err) {
    setHTML('temp_files_status', `<span style="color:red;">Error: ${err}</span>`);
  }
}

// Initialize on load
window.addEventListener('load', () => {
  refreshFolders();
});
