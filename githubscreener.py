def display_paginated_charts(
    results_list, key_prefix="screener", run_mode="Long", run_cap="Mid"
):
  if not results_list:
    st.info(
        "No tickers found or run a scan first using the sidebar execution"
        " engine."
    )
    return

  st.success(
      f"Found {len(results_list)} structurally strong {run_cap} Cap matches!"
  )

  CHARTS_PER_PAGE = 4
  total_pages = max(1, (len(results_list) - 1) // CHARTS_PER_PAGE + 1)

  page_state_key = f"page_num_{key_prefix}_{run_cap.lower()}_{run_mode.lower()}"
  if page_state_key not in st.session_state:
    st.session_state[page_state_key] = 1

  if st.session_state[page_state_key] > total_pages:
    st.session_state[page_state_key] = total_pages

  # --- HELPER TO RENDER CONTROLS BAR (USED AT TOP & BOTTOM) ---
  def render_pagination_bar(location_tag):
    b_col1, b_col2, b_col3 = st.columns([1, 2, 1])

    with b_col1:
      if st.button(
          "⬅️ Prev",
          key=f"prev_{location_tag}_{key_prefix}_{run_cap.lower()}_{run_mode.lower()}",
          use_container_width=True,
      ):
        if st.session_state[page_state_key] > 1:
          st.session_state[page_state_key] -= 1
          st.rerun()

    with b_col2:
      # Allows direct page number entry / stepping while staying synced
      new_page = st.number_input(
          f"Page (of {total_pages})",
          min_value=1,
          max_value=total_pages,
          value=int(st.session_state[page_state_key]),
          step=1,
          key=f"num_{location_tag}_{key_prefix}_{run_cap.lower()}_{run_mode.lower()}",
          label_visibility="collapsed",
      )
      if new_page != st.session_state[page_state_key]:
        st.session_state[page_state_key] = new_page
        st.rerun()

    with b_col3:
      if st.button(
          "Next ➡️",
          key=f"next_{location_tag}_{key_prefix}_{run_cap.lower()}_{run_mode.lower()}",
          use_container_width=True,
      ):
        if st.session_state[page_state_key] < total_pages:
          st.session_state[page_state_key] += 1
          st.rerun()

  # --- TOP CONTROLS ---
  st.markdown("---")
  render_pagination_bar("top")

  current_page = st.session_state[page_state_key]
  start_idx = (current_page - 1) * CHARTS_PER_PAGE
  end_idx = start_idx + CHARTS_PER_PAGE
  page_results = results_list[start_idx:end_idx]

  st.markdown(
      f"<p style='text-align: center; color: #888; font-size: 0.85rem;'>Page"
      f" <b>{current_page}</b> of <b>{total_pages}</b> &nbsp;|&nbsp; Showing"
      f" matches <b>{start_idx + 1}</b> to <b>{min(end_idx, len(results_list))}</b>"
      f" of {len(results_list)}</p>",
      unsafe_allow_html=True,
  )

  # --- RENDER CHARTS ---
  for idx in range(0, len(page_results), 3):
    chunk = page_results[idx : idx + 3]
    cols = st.columns(3)
    for col, r in zip(cols, chunk):
      with col:
        st.markdown(f"### 📊 {r.ticker}")
        m_c1, m_c2, m_c3 = st.columns(3)
        m_c1.metric("Close", f"${r.close:.2f}")
        m_c2.metric(
            "RS Edge",
            f"{r.factor_edge:+.2f}"
            if run_mode == "Long"
            else f"{r.factor_edge:.2f}",
        )
        m_c3.metric("Dist 20SMA", f"{r.dist_sma20:+.2f}%")

        render_tv_chart(r.ticker, mode=run_mode)

  # --- BOTTOM CONTROLS ---
  st.markdown("---")
  render_pagination_bar("bottom")
