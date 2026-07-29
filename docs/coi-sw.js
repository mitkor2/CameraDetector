/* COOP/COEP service worker — enables crossOriginIsolation (and therefore
 * SharedArrayBuffer + multi-threaded WASM) on hosts like GitHub Pages that
 * cannot set response headers. Based on the coi-serviceworker pattern.
 * COEP mode is passed by the registering page via ?coep=credentialless|require-corp
 */
'use strict';

const coepMode = new URL(self.location.href).searchParams.get('coep') === 'credentialless'
  ? 'credentialless' : 'require-corp';

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
  const r = event.request;
  if (r.cache === 'only-if-cached' && r.mode !== 'same-origin') return;

  const request = (coepMode === 'credentialless' && r.mode === 'no-cors')
    ? new Request(r, { credentials: 'omit' })
    : r;

  event.respondWith(fetch(request).then((response) => {
    if (response.status === 0) return response;
    const headers = new Headers(response.headers);
    headers.set('Cross-Origin-Embedder-Policy', coepMode);
    if (coepMode !== 'credentialless') headers.set('Cross-Origin-Resource-Policy', 'cross-origin');
    headers.set('Cross-Origin-Opener-Policy', 'same-origin');
    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers,
    });
  }).catch((e) => console.error(e)));
});
