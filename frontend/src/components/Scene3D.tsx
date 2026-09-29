import { useEffect, useRef } from "react";
import * as THREE from "three";

/** stage: 0 idle · 1 searching · 2 finding · 3 checking · 4 error (driven by real pipeline state) */
export default function Scene3D({ stage }: { stage: number }) {
  const host = useRef<HTMLDivElement>(null);
  const stageRef = useRef(stage);
  useEffect(() => { stageRef.current = stage; }, [stage]);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const canvas = document.createElement("canvas");
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: "low-power" });
    } catch { return; }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(canvas);

    const scene = new THREE.Scene();
    const cam = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
    cam.position.z = 13;
    const violet = new THREE.Color("#8b5cf6"), cyan = new THREE.Color("#22d3ee"), rose = new THREE.Color("#f43f5e");

    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const key = new THREE.PointLight(0x8b5cf6, 90, 0, 1.6); key.position.set(5, 4, 6); scene.add(key);

    // Document shards: glass cards with faint text lines
    const geo = new THREE.PlaneGeometry(1.3, 0.85), edges = new THREE.EdgesGeometry(geo);
    const txt = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute([-.45,.2,0,.45,.2,0,-.45,0,0,.3,0,0,-.45,-.2,0,.1,-.2,0], 3));
    const txtMat = new THREE.LineBasicMaterial({ color: violet, transparent: true, opacity: 0.16 });
    const N = 30;
    const shards = Array.from({ length: N }, (_, i) => {
      const g = new THREE.Group();
      const m = new THREE.MeshBasicMaterial({ color: violet, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false });
      const l = new THREE.LineBasicMaterial({ color: violet, transparent: true, opacity: 0.35 });
      g.add(new THREE.Mesh(geo, m), new THREE.LineSegments(edges, l), new THREE.LineSegments(txt, txtMat));
      g.rotation.set((Math.random() - .5) * .8, (Math.random() - .5) * .8, (Math.random() - .5) * .5);
      scene.add(g);
      return { g, m, l, base: new THREE.Vector3((Math.random() - .5) * 20, (Math.random() - .5) * 11, (Math.random() - .5) * 9 - 2), ph: Math.random() * 6.28, hit: i < 6, lift: 0 };
    });

    // Luminous edges between shards
    const linkPos = new Float32Array(N * 6);
    const linkGeo = new THREE.BufferGeometry(); linkGeo.setAttribute("position", new THREE.BufferAttribute(linkPos, 3));
    scene.add(new THREE.LineSegments(linkGeo, new THREE.LineBasicMaterial({ color: violet, transparent: true, opacity: 0.13 })));

    // Dust
    const dust = new Float32Array(240 * 3).map((_, i) => (i % 3 === 2 ? Math.random() * 12 - 4 : (Math.random() - .5) * (i % 3 ? 14 : 24)));
    const dustGeo = new THREE.BufferGeometry(); dustGeo.setAttribute("position", new THREE.BufferAttribute(dust, 3));
    scene.add(new THREE.Points(dustGeo, new THREE.PointsMaterial({ color: 0xcbb8ff, size: 0.035, transparent: true, opacity: 0.55 })));

    // Orb cluster (the logo as a graph)
    const orbs = new THREE.Group(); orbs.position.set(5.4, 1.3, 1);
    const om = new THREE.MeshStandardMaterial({ color: 0x9b7cff, metalness: 0.7, roughness: 0.22, emissive: 0x2a1466 });
    [[0, .5, 0], [-.55, -.35, .2], [.55, -.3, -.2]].forEach(([x, y, z]) => { const s = new THREE.Mesh(new THREE.SphereGeometry(.32, 32, 32), om); s.position.set(x, y, z); orbs.add(s); });
    scene.add(orbs);

    // Pipeline choreography: query pulse + verification ring
    const pulse = new THREE.Mesh(new THREE.RingGeometry(.92, 1, 64), new THREE.MeshBasicMaterial({ color: violet, transparent: true, side: THREE.DoubleSide }));
    pulse.position.set(0, -4, 1);
    const sweep = new THREE.Mesh(new THREE.TorusGeometry(1, .012, 8, 120), new THREE.MeshBasicMaterial({ color: cyan }));
    sweep.position.set(0, .2, 2); sweep.rotation.x = 1.15;
    scene.add(pulse, sweep);

    let raf = 0, t = 0, mx = 0, my = 0, cx = 0, cy = 0;
    const onMove = (e: PointerEvent) => { mx = e.clientX / window.innerWidth - .5; my = e.clientY / window.innerHeight - .5; };
    const resize = () => { const w = el.clientWidth, h = el.clientHeight; renderer.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); };
    const tmp = new THREE.Vector3();

    const tick = () => {
      raf = requestAnimationFrame(tick);
      t += 0.008; const s = stageRef.current;
      cx += (mx - cx) * .04; cy += (my - cy) * .04;
      cam.position.set(cx * 1.8, -cy * 1.2, 13 + window.scrollY * .001); cam.lookAt(0, 0, 0);
      const tint = s === 4 ? rose : s === 3 ? cyan : violet;
      shards.forEach((o, i) => {
        o.lift += ((s >= 2 && s < 4 ? (o.hit ? 1 : -1) : 0) - o.lift) * .06;
        const up = Math.max(o.lift, 0), dn = Math.min(o.lift, 0);
        o.g.position.set(o.base.x + Math.sin(t + o.ph) * .4, o.base.y + Math.cos(t * .8 + o.ph) * .4, o.base.z + up * 4);
        o.g.rotation.y += .0015;
        o.m.opacity = .07 + up * .16 + dn * .05;
        o.l.opacity = .35 + up * .5 + dn * .25;
        o.l.color.lerp(o.hit && s >= 3 ? tint : violet, .08);
        const j = (i * 7 + 3) % N;
        o.g.getWorldPosition(tmp); linkPos.set([tmp.x, tmp.y, tmp.z], i * 6);
        shards[j].g.getWorldPosition(tmp); linkPos.set([tmp.x, tmp.y, tmp.z], i * 6 + 3);
      });
      linkGeo.attributes.position.needsUpdate = true;
      const p = (t * 1.2) % 1;
      pulse.visible = s === 1; pulse.scale.setScalar(.2 + p * 9); (pulse.material as THREE.MeshBasicMaterial).opacity = (1 - p) * .5;
      sweep.visible = s === 3; sweep.rotation.z = t * 3; sweep.scale.setScalar(3 + Math.sin(t * 3) * .4);
      orbs.rotation.y = t * .6; orbs.rotation.x = Math.sin(t * .5) * .2;
      renderer.render(scene, cam);
    };
    const onVis = () => { cancelAnimationFrame(raf); if (!document.hidden) tick(); };

    resize(); tick();
    window.addEventListener("pointermove", onMove, { passive: true });
    window.addEventListener("resize", resize);
    document.addEventListener("visibilitychange", onVis);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("resize", resize);
      document.removeEventListener("visibilitychange", onVis);
      scene.traverse((o) => { const m = o as THREE.Mesh; m.geometry?.dispose(); });
      renderer.dispose(); canvas.remove();
    };
  }, []);

  return <div ref={host} className="scene" aria-hidden="true" />;
}
