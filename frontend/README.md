# Panda Matching Frontend

React + TypeScript + Vite frontend for the Panda Matching API.

## What It Does

- Chat interface backed by `POST /agent/chat`.
- Structured match cards for ranked recommendations.
- Panda catalog powered by `GET /pandas`.
- Panda photos and local visual assets for the product UI.

## Development

Install dependencies:

```bash
npm install
```

Start the FastAPI backend from the repository root:

```bash
uv run panda-matching-api
```

Start the Vite dev server:

```bash
npm run dev
```

The dev server proxies these backend paths to `http://localhost:8000`:

- `/agent`
- `/matches`
- `/pandas`
- `/health`

## Scripts

```bash
npm run dev
npm run build
npm run lint
npm run preview
```
