# ==========================================
# VIEW: LEADERBOARD
# ==========================================
if st.session_state.nav == 'Leaderboard':
    st.markdown("### 🏆 LEADERBOARD")
    
    leaders = [
        {"rank": "🥇", "name": "Priya S.", "eff": "+2.4%", "savings": "₹8,420", "img": "👩🏽", "badge": "MASTER", "streak": "15 days 🔥"},
        {"rank": "🥈", "name": "Rahul K.", "eff": "+1.8%", "savings": "₹6,850", "img": "👨🏽", "badge": "EXPERT", "streak": "12 days 🔥"},
        {"rank": "🥉", "name": "Amit B.", "eff": "+1.1%", "savings": "₹5,230", "img": "👨🏻", "badge": "EXPERT", "streak": "8 days 🔥"},
        {"rank": "4", "name": "Rant K.", "eff": "+0.8%", "savings": "₹4,100", "img": "👩🏻", "badge": "PRO", "streak": "5 days"},
        {"rank": "5", "name": "Antre G.", "eff": "+0.7%", "savings": "₹3,560", "img": "👨🏾", "badge": "PRO", "streak": "3 days"},
        {"rank": "6", "name": "Divya K.", "eff": "+0.5%", "savings": "₹2,890", "img": "👩🏼", "badge": "", "streak": "2 days"},
    ]
    
    # User's Position
    st.markdown("#### 📍 Your Position")
    st.markdown(f"""
    <div class="metric-card">
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div style='text-align: left;'>
                <div class='metric-label'>Your Rank</div>
                <div style='font-size: 28px; font-weight: 700; color: #10b981;'>#12</div>
            </div>
            <div style='text-align: center;'>
                <div class='metric-label'>Weekly Savings</div>
                <div style='font-size: 24px; font-weight: 700;'>₹1,234</div>
            </div>
            <div style='text-align: right;'>
                <div class='metric-label'>Your Efficiency</div>
                <div style='font-size: 24px; font-weight: 700; color: #fbbf24;'>+0.3%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### 🏅 Top Savers This Week")
    
    for idx, leader in enumerate(leaders):
        card_style = "signal-card positive" if idx < 3 else "signal-card"
        medal_color = "#fbbf24" if idx == 0 else "#c0c0c0" if idx == 1 else "#cd7f32" if idx == 2 else "#cbd5e1"
        
        st.markdown(f"""
        <div class="{card_style}">
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div style='display: flex; align-items: center; gap: 12px; flex: 1;'>
                    <div style='font-size: 32px; font-weight: 700; color: {medal_color}; min-width: 40px;'>{leader['rank']}</div>
                    <div style='font-size: 24px;'>{leader['img']}</div>
                    <div>
                        <div style='font-weight: 700; font-size: 16px; color: #ffffff;'>{leader['name']}</div>
                        <div style='font-size: 12px; color: #94a3b8;'>{leader['streak']}</div>
                    </div>
                </div>
                <div style='text-align: right;'>
                    <div style='font-size: 14px; color: #94a3b8; text-transform: uppercase; font-weight: 600;'>Weekly</div>
                    <div style='font-size: 20px; font-weight: 700; color: #10b981; margin-bottom: 4px;'>{leader['savings']}</div>
                    <div style='font-size: 13px; color: #10b981; font-weight: 600;'>{leader['eff']}</div>
                </div>
            </div>
            <div style='display: flex; gap: 6px; margin-top: 10px;'>
                {f"<span style='background: #10b981; color: white; padding: 4px 8px; border-radius: 6px; font-size: 11px; font-weight: 600;'>🎖️ {leader['badge']}</span>" if leader['badge'] else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### 💡 How to Climb the Leaderboard")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style='font-size: 24px;'>📈</div>
            <div class='metric-label'>Track Trends</div>
            <div style='font-size: 12px; color: #cbd5e1; line-height: 1.4;'>Use market analysis to time transfers</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style='font-size: 24px;'>🎯</div>
            <div class='metric-label'>Build Streaks</div>
            <div style='font-size: 12px; color: #cbd5e1; line-height: 1.4;'>Daily activity multiplies savings</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style='font-size: 24px;'>💰</div>
            <div class='metric-label'>Save More</div>
            <div style='font-size: 12px; color: #cbd5e1; line-height: 1.4;'>Higher amounts = better rankings</div>
        </div>
        """, unsafe_allow_html=True)
