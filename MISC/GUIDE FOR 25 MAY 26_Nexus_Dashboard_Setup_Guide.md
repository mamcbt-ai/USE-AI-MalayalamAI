# Nexus Dashboard — Complete Setup Guide
> Your full developer preparation guide. Follow this before building tomorrow.

---

## ① Prerequisites — Check what you have

Open **Terminal** (Mac: ⌘+Space → "Terminal") or **Command Prompt** (Windows: search "cmd")

```bash
# Check Node.js — you need v18 or higher
node -v       # should show v18.x.x or v20.x.x
npm -v        # should show 9.x or 10.x

# Check VS Code
code --version
```

- No Node.js? → Download LTS from https://nodejs.org (free, one click)
- No VS Code? → Download from https://code.visualstudio.com (free)

---

## ② Terminal — Create your project

Run each command one by one:

```bash
# 1. Go to your Desktop and create a Projects folder
cd ~/Desktop                          # Mac/Linux
mkdir Projects && cd Projects

# cd %USERPROFILE%\Desktop            # Windows
# mkdir Projects && cd Projects

# 2. Create the React + Vite project
npm create vite@latest nexus-dashboard -- --template react
cd nexus-dashboard
npm install

# 3. Install Tailwind CSS
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# 4. Install all needed packages
npm install recharts react-router-dom lucide-react

# 5. Open in VS Code
code .

# 6. Run the dev server
npm run dev
# → Opens at http://localhost:5173
```

---

## ③ Config — 2 files to edit

**tailwind.config.js** — replace entire content with:
```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: { extend: {} },
  plugins: [],
}
```

**src/index.css** — delete everything and paste:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

---

## ④ Folder Structure

Create all folders with one command (run inside nexus-dashboard/):
```bash
mkdir -p src/components/layout src/components/dashboard src/components/tables src/components/auth src/pages src/assets
```

Final structure:
```
nexus-dashboard/
├── src/
│   ├── components/
│   │   ├── layout/       ← Sidebar.jsx, Topbar.jsx, Layout.jsx
│   │   ├── dashboard/    ← KPICards.jsx, RevenueChart.jsx, ActivityFeed.jsx
│   │   ├── tables/       ← DataTable.jsx
│   │   └── auth/         ← Login.jsx, Signup.jsx
│   ├── pages/            ← Dashboard.jsx, Analytics.jsx, Users.jsx, Settings.jsx
│   └── assets/           ← images, logos
├── App.jsx
├── index.css
├── tailwind.config.js
└── package.json
```

---

## ⑤ VS Code Extensions to Install

Press **Ctrl+Shift+X** (Windows) or **⌘+Shift+X** (Mac) to open Extensions panel.

Search and install each:

| Extension | Why you need it |
|-----------|----------------|
| Tailwind CSS IntelliSense | Autocomplete for all Tailwind class names |
| ES7+ React Snippets | Type `rfce` → full React component appears |
| Prettier | Auto-formats your code on every save |
| ESLint | Catches errors before you run the code |
| Auto Rename Tag | Rename open tag → close tag updates automatically |
| Color Highlight | Shows colour swatches next to hex codes |

---

## ⑥ Browser Setup — Chrome

Use **Google Chrome** as your dev browser.

**Install these Chrome Extensions:**
- React Developer Tools — inspect React components live
- PixelZoomer — measure spacing and sizes precisely

**Chrome DevTools (press F12):**
- **Elements tab** — inspect and live-edit any UI element's CSS
- **Console tab** — see errors (red = fix immediately)
- **Network tab** — see all API calls, watch for failures
- **Responsive mode** — phone icon → test any screen size

**Your dev URL:** `http://localhost:5173`

> ⚠️ localhost links only work on YOUR computer. Don't share them.

---

## ⑦ Tomorrow's Build Plan

| Status | Brick | What's in it |
|--------|-------|-------------|
| ✅ Done | Brick 1 — Layout Shell | Dark sidebar, topbar, collapsible nav |
| ✅ Done | Brick 2 — Dashboard Home | KPI cards, chart, activity feed, goals |
| ▶ Tomorrow | Brick 3 — Data Table | Search, sort, filter, pagination |
| ⏳ Queued | Brick 4 — Auth Screens | Login, signup, validation |
| ⏳ Queued | Brick 5 — Settings Page | Profile, notifications, security, billing |
| 🚀 Final | Deploy | Wire routing + deploy to Vercel (free) |

---

## Quick Start Checklist for Tomorrow Morning

- [ ] Node.js installed and `node -v` shows v18+
- [ ] Project created: `nexus-dashboard/` folder exists
- [ ] `npm install` done (node_modules folder present)
- [ ] Tailwind configured
- [ ] VS Code extensions installed
- [ ] `npm run dev` runs and browser opens at localhost:5173
- [ ] Say "let's continue building" → jump straight to Brick 3!

---
*Prepared by Claude · Nexus Dashboard Project · May 2026*
