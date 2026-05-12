interface ServingCommandPanelProps {
  command: string;
  usesPlaceholderModel: boolean;
}

export const ServingCommandPanel = ({
  command,
  usesPlaceholderModel,
}: ServingCommandPanelProps) => {
  return (
    <div className="serving-command">
      <h3>vLLM Starter Command</h3>
      <pre>{command}</pre>
      {usesPlaceholderModel && (
        <p>Replace the model placeholder with the actual model ID or path.</p>
      )}
    </div>
  );
};
