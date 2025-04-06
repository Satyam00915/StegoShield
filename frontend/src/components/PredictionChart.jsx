import { PieChart, Pie, Cell, Tooltip, Legend } from "recharts";

const COLORS = ["#EF4444", "#10B981"];

const PredictionChart = ({ result, confidence }) => {
  const data = [
    { name: "Malicious", value: result === "Malicious" ? confidence : 1 - confidence },
    { name: "Safe", value: result === "Safe" ? confidence : 1 - confidence },
  ];

  return (
    <div className="flex justify-center mt-4">
      <PieChart width={250} height={200}>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          labelLine={false}
          outerRadius={70}
          dataKey="value"
          label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
        >
          {data.map((_, index) => (
            <Cell key={index} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip />
        <Legend />
      </PieChart>
    </div>
  );
};

export default PredictionChart;
