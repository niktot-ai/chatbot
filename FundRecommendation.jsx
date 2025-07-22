import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

const colors = {
  equity: "bg-blue-50 border-blue-300",
  hybrid: "bg-yellow-50 border-yellow-300",
  debt: "bg-green-50 border-green-300"
};

export default function FundRecommendation() {
  return (
    <div className="space-y-4">
      {["equity", "hybrid", "debt"].map((category) =>
        props[category]?.length ? (
          <div key={category}>
            <h2 className="text-xl font-bold capitalize mb-2">
              {category} Funds
            </h2>
            <div className="grid md:grid-cols-2 gap-4">
              {props[category].map((fund, idx) => (
                <Card
                  key={idx}
                  className={`p-4 rounded-2xl border shadow-sm hover:shadow-md transition-all ${colors[category]}`}
                >
                  <div className="flex justify-between items-center">
                    <h3 className="font-semibold">{fund.scheme_name}</h3>
                    <Badge>{fund.Category}</Badge>
                  </div>
                  <Separator className="my-2" />
                  <div className="text-sm">
                    <p><strong>Allocation:</strong> ₹{fund.allocation_amount} ({fund.allocation_percent}%)</p>
                    <p><strong>Returns:</strong> 1Y: {fund.OneYearReturns}%, 3Y: {fund.ThreeYearReturns}%, 5Y: {fund.FiveYearReturns}%</p>
                    <p><strong>Rationale:</strong> {fund.rationale}</p>
                    <p><strong>AMC:</strong> {fund.Encrypt_SchemeCode}</p>
                  </div>
                </Card>
              ))}
            </div>
          </div>
        ) : null
      )}
    </div>
  );
}
