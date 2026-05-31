"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

const API_URL = "/api";

const PLANS = [
  { id: "basic",     name: "Basic",     price: 99,  badge: "",             recordings: "30 recordings/day",    description: "Great for regular use",   color: "#4CAF50" },
  { id: "pro",       name: "Pro",       price: 249, badge: "Most Popular", recordings: "100 recordings/day",   description: "Best for professionals",  color: "#2196F3" },
  { id: "unlimited", name: "Unlimited", price: 499, badge: "",             recordings: "Unlimited recordings", description: "For power users & teams",  color: "#9C27B0" },
];

export default function PricingPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(null);
  const [message, setMessage] = useState("");
  const [token,   setToken]   = useState(null);
  const [myPlan,  setMyPlan]  = useState(null);

  useEffect(() => {
    const stored = localStorage.getItem("token");
    if (!stored) { router.push("/"); return; }
    setToken(stored);
    fetch(`${API_URL}/payment/my-plan`, { headers: { Authorization: `Bearer ${stored}` } })
      .then(r => r.json()).then(d => setMyPlan(d.plan)).catch(() => {});
    if (!document.getElementById("rzp-script")) {
      const s = document.createElement("script");
      s.id = "rzp-script"; s.src = "https://checkout.razorpay.com/v1/checkout.js"; s.async = true;
      document.body.appendChild(s);
    }
  }, []);

  const handleBuyNow = async (plan) => {
    if (!token) { router.push("/"); return; }
    setLoading(plan.id); setMessage("");
    try {
      const res = await fetch(`${API_URL}/payment/create-order`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({ plan_id: plan.id }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to create order");
      const options = {
        key: data.key_id, amount: data.amount, currency: data.currency,
        name: "Malayalam Voice AI", description: `${plan.name} - ${plan.recordings}`,
        order_id: data.order_id,
        handler: async (response) => {
          const vRes = await fetch(`${API_URL}/payment/verify`, {
            method: "POST",
            headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
            body: JSON.stringify({ razorpay_order_id: response.razorpay_order_id, razorpay_payment_id: response.razorpay_payment_id, razorpay_signature: response.razorpay_signature, plan_id: plan.id }),
          });
          const vData = await vRes.json();
          if (vData.success) { setMyPlan(plan.id); setMessage("Payment successful! You are now on the " + plan.name + " Plan."); setTimeout(() => router.push("/"), 2500); }
          else { setMessage("Payment verification failed. Please contact support."); }
        },
        prefill: { email: localStorage.getItem("user_email") || "" },
        theme: { color: plan.color }, modal: { ondismiss: () => setLoading(null) },
      };
      new window.Razorpay(options).open();
    } catch (err) { setMessage("Error: " + err.message); setLoading(null); }
  };

  return (
    <div style={{ minHeight:"100vh", backgroundColor:"#0f0f0f", color:"#fff", padding:"48px 20px", fontFamily:"sans-serif", textAlign:"center" }}>
      <h1 style={{ fontSize:"2.2rem", marginBottom:"8px", fontWeight:800 }}>Choose Your Plan</h1>
      <p style={{ color:"#aaa", marginBottom:"40px" }}>Free: 10 recordings/day · Upgrade for more</p>
      {message && <div style={{ background:"#1e1e1e", border:"1px solid #444", borderRadius:"10px", padding:"14px 20px", marginBottom:"28px", maxWidth:"600px", margin:"0 auto 28px" }}>{message}</div>}
      <div style={{ display:"flex", gap:"24px", justifyContent:"center", flexWrap:"wrap", maxWidth:"960px", margin:"0 auto" }}>
        {PLANS.map((plan) => {
          const isCurrent = myPlan === plan.id;
          return (
            <div key={plan.id} style={{ background:"#1a1a1a", border:"2px solid "+(isCurrent?plan.color:"#333"), borderRadius:"18px", padding:"36px 24px", width:"260px", position:"relative", textAlign:"center" }}>
              {plan.badge && !isCurrent && <div style={{ position:"absolute", top:"-13px", left:"50%", transform:"translateX(-50%)", background:"#2196F3", color:"#fff", padding:"4px 14px", borderRadius:"20px", fontSize:"12px", fontWeight:"bold" }}>{plan.badge}</div>}
              {isCurrent && <div style={{ position:"absolute", top:"-13px", left:"50%", transform:"translateX(-50%)", background:plan.color, color:"#fff", padding:"4px 14px", borderRadius:"20px", fontSize:"12px", fontWeight:"bold" }}>Your Plan</div>}
              <h2 style={{ fontSize:"1.5rem", marginBottom:"16px", marginTop:"8px" }}>{plan.name}</h2>
              <div style={{ marginBottom:"12px" }}>
                <span style={{ fontSize:"1.2rem", verticalAlign:"top", marginTop:"8px", display:"inline-block" }}>Rs.</span>
                <span style={{ fontSize:"3rem", fontWeight:800 }}>{plan.price}</span>
                <span style={{ fontSize:"0.9rem", color:"#aaa" }}>/month</span>
              </div>
              <p style={{ color:plan.color, fontWeight:"bold", marginBottom:"6px" }}>{plan.recordings}</p>
              <p style={{ color:"#aaa", fontSize:"0.85rem", marginBottom:"28px" }}>{plan.description}</p>
              <button
                style={{ width:"100%", padding:"14px", border:"none", borderRadius:"10px", backgroundColor:isCurrent?"#333":plan.color, color:"#fff", fontSize:"1rem", fontWeight:"bold", cursor:isCurrent?"default":"pointer", opacity:loading===plan.id?0.7:1 }}
                onClick={() => !isCurrent && handleBuyNow(plan)}
                disabled={loading===plan.id||isCurrent}
              >{isCurrent?"Current Plan":loading===plan.id?"Processing...":"Buy Now Rs."+plan.price}</button>
            </div>
          );
        })}
      </div>
      <p style={{ color:"#555", marginTop:"32px", fontSize:"0.82rem" }}>Secure payment via Razorpay · Cancel anytime</p>
    </div>
  );
}



