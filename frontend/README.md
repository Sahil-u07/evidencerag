# EvidenceRAG frontend

React, TypeScript and Vite, with plain CSS and no runtime dependencies beyond React.

## Setup

```
npm ci
npm run dev
```

The app opens at http://localhost:5173. It needs the API running. From `backend/`, run `uvicorn app.main:app`. The API only accepts requests from port 5173, so keep that port.

## Environment

`VITE_API_BASE` is the API's base URL. The default is `http://127.0.0.1:8000`.

## Scripts

- `npm run dev` starts the dev server.
- `npm run build` type-checks and builds to `dist/`.
- `npm run lint` runs ESLint.
- `npm test` runs the unit tests in `src/lib` with Node's test runner.

## Demo mode

The "Demo data" checkbox in the left column shows the app without a backend. The data is made up, a banner stays on screen, and answers never get the Verified verdict.
