// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import { loadMesh, loadMeshSet } from '../../mesh-loader';
import {
  loadWebModel,
  materialFor,
  primitiveGeometry
} from '../../web-model';

export interface BodyTreeVisualization {
  destroy(): void;
}

interface MeshMetrics {
  bytes: number;
  triangles: number;
  vertices: number;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) { return `${bytes} B`; }
  if (bytes < 1024 ** 2) { return `${(bytes / 1024).toFixed(1)} KB`; }
  return `${(bytes / 1024 ** 2).toFixed(2)} MB`;
}

function formatCount(count: number): string {
  return new Intl.NumberFormat('en', { notation: 'compact' }).format(count);
}

function metricsLabel(metrics: MeshMetrics): string {
  const source = metrics.bytes === 0 ? 'generated' : formatBytes(metrics.bytes);
  return `${formatCount(metrics.triangles)} tris · ${source}`;
}

function requiredElement(root: ParentNode, selector: string): HTMLElement {
  const element = root.querySelector<HTMLElement>(selector);
  if (!element) { throw new Error(`Missing required element ${selector}`); }
  return element;
}

function setQuaternion(
  object: THREE.Object3D,
  [w, x, y, z]: [number, number, number, number]
): void {
  object.quaternion.set(x, y, z, w);
}

export async function initializeBodyTreeVisualization(
  parent: HTMLElement,
  modelBasePath = '/so101_assets',
  modelUrl = '/so101.json'
): Promise<BodyTreeVisualization> {
  const model = await loadWebModel(modelUrl);
  const bodiesWithVisuals = model.bodies.map(body => ({
    ...body,
    visuals: body.geometries.filter(geometry => geometry.role === 'visual')
  }));
  const visualCount = bodiesWithVisuals.reduce((n, body) => n + body.visuals.length, 0);
  const root = document.createElement('div');
  root.className = 'body-tree-root';
  root.innerHTML = `
    <aside class="body-tree-sidebar">
      <header>
        <strong>Geometry complexity</strong>
        <span>${bodiesWithVisuals.length} bodies ·
          ${visualCount} geoms</span>
      </header>
      <input class="body-tree-search" type="search"
        placeholder="Filter bodies and geoms…" aria-label="Filter bodies and geoms">
      <div class="body-tree-list"></div>
    </aside>
    <section class="body-tree-inspector">
      <div class="body-tree-viewport"></div>
      <div class="body-tree-info">
        <strong>Loading geometry diagnostics…</strong>
        <span>Polygon counts include generated primitives and served Meshopt GLBs.</span>
      </div>
    </section>`;
  parent.querySelector('.placeholder')?.replaceWith(root);
  if (!root.parentElement) { parent.appendChild(root); }

  const viewport = requiredElement(root, '.body-tree-viewport');
  const list = requiredElement(root, '.body-tree-list');
  const info = requiredElement(root, '.body-tree-info');
  const search = requiredElement(root, '.body-tree-search') as HTMLInputElement;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.shadowMap.enabled = true;
  viewport.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f8ff);
  const camera = new THREE.PerspectiveCamera(42, 1, 0.001, 100);
  camera.up.set(0, 0, 1);
  camera.position.set(0.48, 0.48, 0.38);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0, 0.1);
  scene.add(new THREE.HemisphereLight(0xddeeff, 0xffffff, 2.2));
  const light = new THREE.DirectionalLight(0xfff2d6, 3);
  light.position.set(2, 2, 5);
  scene.add(light);
  const grid = new THREE.GridHelper(1, 20, 0x9aa9bc, 0xd5dde8);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);

  const selected = new THREE.MeshStandardMaterial({
    color: 0x38bdf8, emissive: 0x075985, roughness: 0.35
  });
  const basePath = modelBasePath.replace(/\/$/, '');
  const bodies = new Map<string, THREE.Group>();
  const meshes = new Map<string, THREE.Mesh>();
  const meshMetrics = new Map<string, MeshMetrics>();
  const bodyMetrics = new Map<string, MeshMetrics>();
  const rows = new Map<string, HTMLElement>();
  const hiddenMeshes = new Set<string>();
  const hiddenBodies = new Set<string>();
  let selectedId = '';

  const select = (id: string, title: string, details: string, object: THREE.Object3D): void => {
    selectedId = id;
    for (const [meshId, mesh] of meshes) {
      mesh.material = meshId === id ? selected : mesh.userData.baseMaterial as THREE.Material;
    }
    for (const [rowId, row] of rows) { row.classList.toggle('selected', rowId === id); }
    info.innerHTML = `<strong>${title}</strong><span>${details}</span>`;
    const box = new THREE.Box3().setFromObject(object);
    if (!box.isEmpty()) {
      const center = box.getCenter(new THREE.Vector3());
      const size = Math.max(box.getSize(new THREE.Vector3()).length(), 0.08);
      controls.target.copy(center);
      camera.position.copy(center).add(new THREE.Vector3(size, size, size * 0.75));
      controls.update();
    }
  };

  for (const link of bodiesWithVisuals) {
    const group = new THREE.Group();
    bodies.set(link.name, group);
    const origin = new THREE.Group();
    origin.position.set(...link.position);
    setQuaternion(origin, link.quaternion);
    origin.add(group);
    const parentBody = link.name !== link.parent ? bodies.get(link.parent) : undefined;
    if (parentBody !== undefined) { parentBody.add(origin); } else { scene.add(origin); }

    const bodyRow = document.createElement('details');
    bodyRow.className = 'body-tree-body';
    bodyRow.open = true;
    const summary = document.createElement('summary');
    const initialLabel = link.visuals.length === 0 ? 'no visuals' : 'loading…';
    summary.innerHTML = `<span class="body-tree-icon">B</span>
      <span>${link.name}</span><small>${initialLabel}</small>
      <button class="body-tree-toggle" type="button" title="Hide body geoms"
        aria-label="Toggle ${link.name} geoms" aria-pressed="false">●</button>`;
    bodyRow.appendChild(summary);
    rows.set(link.name, summary);
    summary.addEventListener('click', () => {
      const joint = link.joints.at(0);
      const metrics = bodyMetrics.get(link.name);
      const hierarchy = joint !== undefined
        ? `joint ${joint.name} · parent ${link.parent}`
        : 'root body';
      select(link.name, link.name, metrics
        ? `${hierarchy} · ${metricsLabel(metrics)} · ${formatCount(metrics.vertices)} vertices`
        : `${hierarchy} · loading mesh diagnostics…`, group);
    });
    const bodyToggle = requiredElement(summary, '.body-tree-toggle') as HTMLButtonElement;
    bodyToggle.addEventListener('click', event => {
      event.preventDefault();
      event.stopPropagation();
      const visible = hiddenBodies.has(link.name);
      if (visible) { hiddenBodies.delete(link.name); } else { hiddenBodies.add(link.name); }
      for (const [meshId, mesh] of meshes) {
        if (meshId.startsWith(`${link.name}/`)) {
          mesh.visible = visible && !hiddenMeshes.has(meshId);
        }
      }
      bodyToggle.classList.toggle('off', !visible);
      bodyToggle.ariaPressed = String(!visible);
      bodyToggle.title = visible ? 'Hide body geoms' : 'Show body geoms';
    });

    link.visuals.forEach((visual, index) => {
      const visualName = visual.mesh ?? visual.name;
      const id = `${link.name}/${index}`;
      const row = document.createElement('button');
      row.className = 'body-tree-geom';
      row.type = 'button';
      row.innerHTML = `<span class="body-tree-icon">G</span>
        <span>${visualName}</span>
        <small>loading…</small>
        <span class="body-tree-toggle" role="switch" tabindex="0"
          title="Hide geom" aria-label="Hide ${visualName}" aria-checked="true">●</span>`;
      bodyRow.appendChild(row);
      rows.set(id, row);
      const addGeometry = (geometry: THREE.BufferGeometry, bytes: number): void => {
        const metrics = {
          bytes,
          triangles: geometry.index
            ? geometry.index.count / 3
            : geometry.getAttribute('position').count / 3,
          vertices: geometry.getAttribute('position').count
        };
        meshMetrics.set(id, metrics);
        const total = bodyMetrics.get(link.name) ?? { bytes: 0, triangles: 0, vertices: 0 };
        total.bytes += metrics.bytes;
        total.triangles += metrics.triangles;
        total.vertices += metrics.vertices;
        bodyMetrics.set(link.name, total);
        requiredElement(row, 'small').textContent = metricsLabel(metrics);
        requiredElement(summary, 'small').textContent = metricsLabel(total);
        const baseMaterial = materialFor(visual, model.materials);
        const mesh = new THREE.Mesh(geometry, baseMaterial);
        mesh.position.set(...visual.position);
        setQuaternion(mesh, visual.quaternion);
        mesh.userData.visual = visual;
        mesh.userData.baseMaterial = baseMaterial;
        group.add(mesh);
        meshes.set(id, mesh);
        mesh.visible = !hiddenBodies.has(link.name) && !hiddenMeshes.has(id);
        if (selectedId === id) { mesh.material = selected; }
        if (meshMetrics.size === visualCount) {
          const all = [...meshMetrics.values()].reduce((sum, item) => ({
            bytes: sum.bytes + item.bytes,
            triangles: sum.triangles + item.triangles,
            vertices: sum.vertices + item.vertices
          }), { bytes: 0, triangles: 0, vertices: 0 });
          info.innerHTML = `<strong>Full robot visual set</strong>
            <span>${metricsLabel(all)} · ${formatCount(all.vertices)} vertices ·
              ${meshMetrics.size} visual geoms</span>`;
        }
      };
      if (visual.type === 'mesh' && visual.mesh !== undefined) {
        const meshName = visual.mesh;
        const meshFile = visual.meshFile;
        // Meshes packed into a shared GLB have no individual file size;
        // attribute each an even share of the pack it was found in as an approximation.
        const geometryLoad = meshFile !== undefined
          ? loadMeshSet(`${basePath}/${meshFile}`).then(({ bytes, geometries }) => {
            const geometry = geometries.get(meshName);
            if (geometry === undefined) {
              throw new Error(`Mesh node "${meshName}" not found in ${meshFile}`);
            }
            return { bytes: bytes / geometries.size, geometry };
          })
          : loadMesh(`${basePath}/${meshName}`);
        geometryLoad.then(({ bytes, geometry }) => {
          addGeometry(geometry, bytes);
        }).catch((error: unknown) => {
          requiredElement(row, 'small').textContent = 'load error';
          console.error(error);
        });
      } else {
        const geometry = primitiveGeometry(visual);
        if (geometry !== undefined) {
          addGeometry(geometry, 0);
        } else {
          requiredElement(row, 'small').textContent = 'unsupported';
        }
      }
      row.addEventListener('click', () => {
        const mesh = meshes.get(id) ?? group;
        const type = visual.type === 'mesh' ? 'mesh' : `${visual.type} primitive`;
        const metrics = meshMetrics.get(id);
        select(id, visualName, metrics
          ? `${type} · ${metricsLabel(metrics)} · ${formatCount(metrics.vertices)} vertices`
          : `${type} · loading mesh diagnostics…`, mesh);
      });
      const geomToggle = requiredElement(row, '.body-tree-toggle');
      const toggleGeom = (event: Event): void => {
        event.preventDefault();
        event.stopPropagation();
        const visible = hiddenMeshes.has(id);
        if (visible) { hiddenMeshes.delete(id); } else { hiddenMeshes.add(id); }
        const mesh = meshes.get(id);
        if (mesh) { mesh.visible = visible && !hiddenBodies.has(link.name); }
        geomToggle.classList.toggle('off', !visible);
        geomToggle.setAttribute('aria-checked', String(visible));
        geomToggle.title = visible ? 'Hide geom' : 'Show geom';
      };
      geomToggle.addEventListener('click', toggleGeom);
      geomToggle.addEventListener('keydown', event => {
        if (event.key === ' ' || event.key === 'Enter') { toggleGeom(event); }
      });
    });
    list.appendChild(bodyRow);
  }

  const filter = (): void => {
    const query = search.value.trim().toLowerCase();
    for (const details of list.querySelectorAll<HTMLElement>('.body-tree-body')) {
      details.hidden = query !== '' && !details.textContent.toLowerCase().includes(query);
    }
  };
  search.addEventListener('input', filter);
  const resize = (): void => {
    const width = viewport.clientWidth || 600;
    const height = viewport.clientHeight || 520;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };
  const observer = new ResizeObserver(resize);
  observer.observe(viewport);
  resize();
  let frame = 0;
  let destroyed = false;
  const animate = (): void => {
    if (destroyed) { return; }
    frame = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  };
  animate();

  return Promise.resolve({ destroy(): void {
    destroyed = true;
    cancelAnimationFrame(frame);
    observer.disconnect();
    controls.dispose();
    renderer.dispose();
    for (const mesh of meshes.values()) {
      (mesh.userData.baseMaterial as THREE.Material | undefined)?.dispose();
    }
    selected.dispose();
    root.remove();
  } });
}
