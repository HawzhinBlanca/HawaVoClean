import { useEffect } from 'react';
import { Actions } from './components/Actions';
import { Footer } from './components/Footer';
import { Header } from './components/Header';
import { MetricsTiles } from './components/MetricsTiles';
import { ProcessButton } from './components/ProcessButton';
import { ProfileControl } from './components/ProfileControl';
import { SourceStrip } from './components/SourceStrip';
import { SpectrumDisplay } from './components/SpectrumDisplay';
import { Transport } from './components/Transport';
import { WaveformDisplay } from './components/WaveformDisplay';
import { connectEngine } from './state/actions';

export default function App() {
  useEffect(() => {
    void connectEngine();
    // Swallow drops outside the drop zone so the shell never navigates away.
    const prevent = (e: DragEvent): void => {
      e.preventDefault();
    };
    window.addEventListener('dragover', prevent);
    window.addEventListener('drop', prevent);
    return () => {
      window.removeEventListener('dragover', prevent);
      window.removeEventListener('drop', prevent);
    };
  }, []);

  return (
    <div className="app">
      <Header />
      <SourceStrip />
      <main className="main">
        <WaveformDisplay />
        <aside className="right">
          <section className="panel spectrum-panel">
            <SpectrumDisplay />
            <MetricsTiles />
          </section>
          <section className="panel controls">
            <ProfileControl />
            <ProcessButton />
            <Transport />
            <Actions />
          </section>
        </aside>
      </main>
      <Footer />
    </div>
  );
}
