import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles/app.css';
// After app.css on purpose: interaction.css is the behaviour layer and has to
// be able to answer a resting style the design system set.
import './styles/interaction.css';

const root = document.getElementById('root');
if (!root) throw new Error('#root missing');
createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
