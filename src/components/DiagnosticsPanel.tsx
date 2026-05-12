import { type DiagnosticMessage } from '../types';

interface DiagnosticsPanelProps {
  diagnostics: DiagnosticMessage[];
}

export const DiagnosticsPanel = ({ diagnostics }: DiagnosticsPanelProps) => {
  return (
    <div className="diagnostics-panel">
      <h3>Diagnostics</h3>
      <ul>
        {diagnostics.map((diagnostic) => (
          <li
            key={diagnostic.id}
            className={`diagnostic diagnostic-${diagnostic.severity}`}
          >
            <strong>{diagnostic.title}</strong>
            <span>{diagnostic.detail}</span>
            <em>{diagnostic.action}</em>
          </li>
        ))}
      </ul>
    </div>
  );
};
