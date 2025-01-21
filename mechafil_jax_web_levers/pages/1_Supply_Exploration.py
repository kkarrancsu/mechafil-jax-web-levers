#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union

from datetime import date, timedelta

import time

import numpy as np
import pandas as pd
import jax.numpy as jnp

import streamlit as st
import streamlit.components.v1 as components
import st_debug as d
import altair as alt

# import pystarboard.data as psb

import mechafil_jax.data as data
import mechafil_jax.sim as sim
import mechafil_jax.constants as C
import mechafil_jax.minting as minting
import mechafil_jax.date_utils as du

import scenario_generator.utils as u

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# local_css("debug.css")

def create_gamma_vector(upgrade_date, forecast_length, current_date, ramp_len_days=90):
    gamma_smooth = np.ones(forecast_length)
    update_day = (upgrade_date-current_date).days

    ramp_gamma = np.linspace(1, 0.7, ramp_len_days)
    
    ramp_start_idx = update_day
    ramp_end_idx = min(forecast_length, ramp_start_idx + ramp_len_days)
    
    print(upgrade_date, current_date, forecast_length, ramp_len_days, update_day, ramp_start_idx, ramp_end_idx)
    
    gamma_smooth[ramp_start_idx:ramp_end_idx] = ramp_gamma[0:(ramp_end_idx-ramp_start_idx)]
    gamma_smooth[ramp_end_idx:] = 0.7
    return gamma_smooth

@st.cache_data
def get_offline_data(start_date, current_date, end_date):
    PUBLIC_AUTH_TOKEN='Bearer ghp_EviOPunZooyAagPPmftIsHfWarumaFOUdBUZ'
    offline_data = data.get_simulation_data(PUBLIC_AUTH_TOKEN, start_date, current_date, end_date)

    _, hist_rbp = u.get_historical_daily_onboarded_power(current_date-timedelta(days=180), current_date)
    _, hist_rr = u.get_historical_renewal_rate(current_date-timedelta(days=180), current_date)
    _, hist_fpr = u.get_historical_filplus_rate(current_date-timedelta(days=180), current_date)

    smoothed_last_historical_rbp = float(np.median(hist_rbp[-30:]))
    smoothed_last_historical_rr = float(np.median(hist_rr[-30:]))
    smoothed_last_historical_fpr = float(np.median(hist_fpr[-30:]))

    # psb.setup_spacescope(PUBLIC_AUTH_TOKEN)
    # hist_df = psb.get_historical_network_stats(start_date-timedelta(days=180), current_date, current_date)
    # print(hist_df)

    return offline_data, smoothed_last_historical_rbp, smoothed_last_historical_rr, smoothed_last_historical_fpr

def plot_panel(scenario_results, baseline, start_date, current_date, end_date):
    # convert results dictionary into a dataframe so that we can use altair to make nice plots
    status_quo_results = scenario_results['status-quo']
    configured_results = scenario_results['configured']
    
    col1, col2, col3, col4 = st.columns(4)
    
    power_dff = pd.DataFrame()
    power_dff['RBP'] = status_quo_results['rb_total_power_eib']
    power_dff['QAP'] = status_quo_results['qa_total_power_eib']
    power_dff['Baseline'] = baseline
    power_dff['date'] = pd.to_datetime(du.get_t(start_date, end_date=end_date))

    minting_dff = pd.DataFrame()
    minting_dff['StatusQuo'] = status_quo_results['day_network_reward']
    minting_dff['Configured'] = configured_results['day_network_reward']
    minting_dff['date'] = pd.to_datetime(du.get_t(start_date, end_date=end_date))

    cs_dff = pd.DataFrame()
    cs_dff['StatusQuo'] = status_quo_results['circ_supply'] / 1e6
    cs_dff['Configured'] = configured_results['circ_supply'] / 1e6
    cs_dff['date'] = pd.to_datetime(du.get_t(start_date, end_date=end_date))

    cs_delta_dff = pd.DataFrame()
    cs_sq = np.diff(np.asarray(status_quo_results['circ_supply']) / 1e6)
    cs_conf = np.diff(np.asarray(configured_results['circ_supply']) / 1e6)
    cs_delta_dff['StatusQuo'] = cs_sq[1:]
    cs_delta_dff['Configured'] = cs_conf[1:]
    cs_delta_dff['date'] = pd.to_datetime(du.get_t(start_date+timedelta(days=2), end_date=end_date))

    as_dff = pd.DataFrame()
    as_dff['StatusQuo'] = status_quo_results['available_supply'] / 1e6
    as_dff['Configured'] = configured_results['available_supply'] / 1e6
    as_dff['date'] = pd.to_datetime(du.get_t(start_date, end_date=end_date))
    as_dff = as_dff[1:]

    locked_dff = pd.DataFrame()
    locked_dff['StatusQuo'] = (status_quo_results['network_locked'] / 1e6)
    locked_dff['Configured'] = (configured_results['network_locked'] / 1e6)
    locked_dff['StatusQuo'] = locked_dff['StatusQuo'].rolling(window=180, min_periods=1).mean()
    locked_dff['Configured'] = locked_dff['Configured'].rolling(window=180, min_periods=1).mean()
    locked_dff['date'] = pd.to_datetime(du.get_t(start_date, end_date=end_date))

    pledge_dff = pd.DataFrame()
    pledge_dff['StatusQuo'] = status_quo_results['day_pledge_per_QAP']
    pledge_dff['Configured'] = configured_results['day_pledge_per_QAP']
    pledge_dff['date'] = pd.to_datetime(du.get_t(start_date, forecast_length=pledge_dff.shape[0]))

    roi_dff = pd.DataFrame()
    roi_dff['StatusQuo'] = status_quo_results['1y_sector_roi'] * 100
    roi_dff['Configured'] = configured_results['1y_sector_roi'] * 100
    roi_dff['date'] = pd.to_datetime(du.get_t(start_date, forecast_length=roi_dff.shape[0]))

    rps_dff = pd.DataFrame()
    rps_dff['StatusQuo'] = status_quo_results['1y_return_per_sector'] * 100
    rps_dff['Configured'] = configured_results['1y_return_per_sector'] * 100
    rps_dff['date'] = pd.to_datetime(du.get_t(start_date, forecast_length=roi_dff.shape[0]))

    drps_dff = pd.DataFrame()
    drps_dff['StatusQuo'] = status_quo_results['day_rewards_per_sector'] * 100
    drps_dff['Configured'] = configured_results['day_rewards_per_sector'] * 100
    drps_dff['date'] = pd.to_datetime(du.get_t(start_date, end_date=end_date))

    dnr_dff = pd.DataFrame()
    dnr_dff['StatusQuo'] = status_quo_results['day_network_reward'] * 100
    dnr_dff['Configured'] = configured_results['day_network_reward'] * 100
    dnr_dff['date'] = pd.to_datetime(du.get_t(start_date, end_date=end_date))

    today = date.today()
    today_line = alt.Chart(pd.DataFrame({'date': [today]})).mark_rule(color='grey', strokeDash=[5, 5]).encode(
        x='date:T'
    )
    with col1:
        power_df = pd.melt(power_dff, id_vars=["date"], 
                           value_vars=[
                               "Baseline", 
                               "RBP", "QAP",],
                           var_name='Power', 
                           value_name='EIB')
        power_df['EIB'] = power_df['EIB']
        power = (
            alt.Chart(power_df)
            .mark_line()
            .encode(x=alt.X("date", title="", axis=alt.Axis(format="%b %Y", labelAngle=-45, tickCount=5)), 
                    y=alt.Y("EIB").scale(type='log'), color=alt.Color('Power', legend=alt.Legend(orient="top", title=None)))
            .properties(title="Network Power")
        )
        final_power = (power + today_line).configure_title(fontSize=14, anchor='middle')
        st.altair_chart(final_power.interactive(), use_container_width=True) 

        # NOTE: adding the tooltip here causes the chart to not render for some reason
        # Following the directions here: https://docs.streamlit.io/library/api-reference/charts/st.altair_chart
        roi_df = pd.melt(roi_dff, id_vars=["date"], 
                         value_vars=["StatusQuo", "Configured"], #, "Optimistic"], 
                         var_name='Scenario', 
                         value_name='%')
        roi = (
            alt.Chart(roi_df)
            .mark_line()
            .encode(x=alt.X("date", title="", axis=alt.Axis(format="%b %Y", labelAngle=-45, tickCount=5)), 
                    y=alt.Y("%"), color=alt.Color('Scenario', legend=None))
            .properties(title="1Y Sector FoFR")
        )
        final_roi = (roi + today_line).configure_title(fontSize=14, anchor='middle')
        st.altair_chart(final_roi.interactive(), use_container_width=True)

        # rps_df = pd.melt(rps_dff, id_vars=["date"], 
        #                  value_vars=["StatusQuo", "Configured"], #, "Optimistic"], 
        #                  var_name='Scenario', 
        #                  value_name='%')
        # rps = (
        #     alt.Chart(rps_df)
        #     .mark_line()
        #     .encode(x=alt.X("date", title="", axis=alt.Axis(labelAngle=-45)), 
        #             y=alt.Y("%"), color=alt.Color('Scenario', legend=None))
        #     .properties(title="1Y Return per Sector")
        #     .configure_title(fontSize=14, anchor='middle')
        # )
        # st.altair_chart(rps.interactive(), use_container_width=True)

    with col2:
        pledge_per_qap_df = pd.melt(pledge_dff, id_vars=["date"],
                                    value_vars=["StatusQuo", "Configured"], #, "Optimistic"], 
                                    var_name='Scenario', value_name='FIL')
        day_pledge_per_QAP = (
            alt.Chart(pledge_per_qap_df)
            .mark_line()
            .encode(x=alt.X("date", title="", axis=alt.Axis(format="%b %Y", labelAngle=-45, tickCount=5)), 
                    y=alt.Y("FIL"), color=alt.Color('Scenario', legend=alt.Legend(orient="top", title=None)))
            .properties(title="Pledge/32GiB QAP")
        )
        final_day_pledge_per_QAP = (day_pledge_per_QAP + today_line).configure_title(fontSize=14, anchor='middle')
        st.altair_chart(final_day_pledge_per_QAP.interactive(), use_container_width=True)

        minting_df = pd.melt(minting_dff, id_vars=["date"],
                             value_vars=["StatusQuo", "Configured"], #, "Optimistic"], 
                             var_name='Scenario', value_name='FILRate')
        minting = (
            alt.Chart(minting_df)
            .mark_line()
            .encode(x=alt.X("date", title="", axis=alt.Axis(format="%b %Y", labelAngle=-45, tickCount=5)), 
                    y=alt.Y("FILRate", title='FIL/day'), color=alt.Color('Scenario', legend=None))
            .properties(title="Minting Rate")
        )
        final_minting = (minting + today_line).configure_title(fontSize=14, anchor='middle')
        st.altair_chart(final_minting.interactive(), use_container_width=True)

        # drps_df = pd.melt(drps_dff, id_vars=["date"], 
        #                  value_vars=["StatusQuo", "Configured"], #, "Optimistic"], 
        #                  var_name='Scenario', 
        #                  value_name='%')
        # drps = (
        #     alt.Chart(drps_df)
        #     .mark_line()
        #     .encode(x=alt.X("date", title="", axis=alt.Axis(labelAngle=-45)), 
        #             y=alt.Y("%"), color=alt.Color('Scenario', legend=None))
        #     .properties(title="Day Return per Sector")
        #     .configure_title(fontSize=14, anchor='middle')
        # )
        # st.altair_chart(drps.interactive(), use_container_width=True)

    with col3:
        cs_df = pd.melt(cs_dff, id_vars=["date"],
                             value_vars=["StatusQuo", "Configured"], #, "Optimistic"], 
                             var_name='Scenario', value_name='cs')
        cs = (
            alt.Chart(cs_df)
            .mark_line()
            .encode(x=alt.X("date", title="", axis=alt.Axis(format="%b %Y", labelAngle=-45, tickCount=5)), 
                    y=alt.Y("cs", title='M-FIL'), color=alt.Color('Scenario', legend=None))
            .properties(title="Circulating Supply")
        )
        final_cs = (cs + today_line).configure_title(fontSize=14, anchor='middle')
        st.altair_chart(final_cs.interactive(), use_container_width=True)

        locked_df = pd.melt(locked_dff, id_vars=["date"],
                             value_vars=["StatusQuo", "Configured"], #, "Optimistic"], 
                             var_name='Scenario', value_name='cs')
        locked = (
            alt.Chart(locked_df)
            .mark_line()
            .encode(x=alt.X("date", title="", axis=alt.Axis(format="%b %Y", labelAngle=-45, tickCount=5)), 
                    y=alt.Y("cs", title='M-FIL'), color=alt.Color('Scenario', legend=None))
            .properties(title="Network Locked")
        )
        final_locked = (locked + today_line).configure_title(fontSize=14, anchor='middle')
        st.altair_chart(final_locked.interactive(), use_container_width=True)

        # dnr_df = pd.melt(dnr_dff, id_vars=["date"], 
        #                  value_vars=["StatusQuo", "Configured"], #, "Optimistic"], 
        #                  var_name='Scenario', 
        #                  value_name='%')
        # dnr = (
        #     alt.Chart(dnr_df)
        #     .mark_line()
        #     .encode(x=alt.X("date", title="", axis=alt.Axis(labelAngle=-45)), 
        #             y=alt.Y("%"), color=alt.Color('Scenario', legend=None))
        #     .properties(title="Day Network Reward")
        #     .configure_title(fontSize=14, anchor='middle')
        # )
        # st.altair_chart(dnr.interactive(), use_container_width=True)

    with col4:
        csd_df = pd.melt(cs_delta_dff, id_vars=["date"],
                             value_vars=["StatusQuo", "Configured"], #, "Optimistic"], 
                             var_name='Scenario', value_name='cs')
        csd = (
            alt.Chart(csd_df)
            .mark_line()
            .encode(x=alt.X("date", title="", axis=alt.Axis(format="%b %Y", labelAngle=-45, tickCount=5)), 
                    y=alt.Y("cs", title='M-FIL'), color=alt.Color('Scenario', legend=None))
            .properties(title="Circ Supply - Delta")
        )
        final_csd = (csd + today_line).configure_title(fontSize=14, anchor='middle')
        st.altair_chart(final_csd.interactive(), use_container_width=True)
        
        as_df = pd.melt(as_dff, id_vars=["date"],
                             value_vars=["StatusQuo", "Configured"], #, "Optimistic"], 
                             var_name='Scenario', value_name='cs')
        aas = (
            alt.Chart(as_df)
            .mark_line()
            .encode(x=alt.X("date", title="", axis=alt.Axis(format="%b %Y", labelAngle=-45, tickCount=5)), 
                    y=alt.Y("cs", title='M-FIL'), color=alt.Color('Scenario', legend=None))
            .properties(title="Available Supply")
        )
        final_aas = (aas + today_line).configure_title(fontSize=14, anchor='middle')
        st.altair_chart(final_aas.interactive(), use_container_width=True)

        # locked_df = pd.melt(locked_dff, id_vars=["date"],
        #                      value_vars=["StatusQuo", "Configured"], #, "Optimistic"], 
        #                      var_name='Scenario', value_name='cs')
        # locked = (
        #     alt.Chart(locked_df)
        #     .mark_line()
        #     .encode(x=alt.X("date", title="", axis=alt.Axis(format="%b %Y", labelAngle=-45, tickCount=5)), 
        #             y=alt.Y("cs", title='M-FIL'), color=alt.Color('Scenario', legend=None))
        #     .properties(title="Network Locked")
        # )
        # final_locked = (locked + today_line).configure_title(fontSize=14, anchor='middle')
        # st.altair_chart(final_locked.interactive(), use_container_width=True)


def add_costs(results_dict, cost_scaling_constant=0.1, filp_scaling_cost_pct=0.5):
    # (returns*multiplier - cost)/(pledge*multiplier)
    # TODO: allow user to configure these within reasonable bounds
    # cost_scaling_constant = 0.1
    # filp_scaling_cost_pct = 0.5

    # compute costs for the FIL+ case
    multiplier = 10
    rps = results_dict['1y_return_per_sector']
    dppq = results_dict['day_pledge_per_QAP'][0:len(rps)]
    
    filp_roi_scaling_costs = dppq*multiplier*cost_scaling_constant
    filp_roi_total_costs = filp_roi_scaling_costs/filp_scaling_cost_pct
    roi_fixed_costs = filp_roi_total_costs - filp_roi_scaling_costs
    results_dict['FIL+'] = 100*(rps*multiplier - filp_roi_total_costs)/(dppq*multiplier)

    # relative to FIL+, compute costs for the CC case
    multiplier = 1
    cc_roi_scaling_costs = dppq*multiplier*cost_scaling_constant
    cc_roi_total_costs = cc_roi_scaling_costs + roi_fixed_costs
    results_dict['CC'] = 100*(rps*multiplier - cc_roi_total_costs)/(dppq*multiplier)
    return results_dict

def run_sim(rbp, rr, fpr, lock_target, start_date, current_date, forecast_length_days, sector_duration_days, offline_data, 
            cost_scaling_constant=0.1, filp_scaling_cost_pct=0.5):
    simulation_results = sim.run_sim(
        rbp,
        rr,
        fpr,
        lock_target,

        start_date,
        current_date,
        forecast_length_days,
        sector_duration_days,
        offline_data
    )
    return simulation_results #, yearly_returns_df

def forecast_economy(start_date=None, current_date=None, end_date=None, forecast_length_days=365*6):
    t1 = time.time()
    
    rb_onboard_power_pib_day =  st.session_state['rbp_slider']
    renewal_rate_pct = st.session_state['rr_slider']
    fil_plus_rate_pct = st.session_state['fpr_slider']
    lock_target = st.session_state['lock_target_slider']
    sector_duration_days = st.session_state['sector_duration_slider']
    
    # get offline data
    t2 = time.time()
    offline_data, _, _, _ = get_offline_data(start_date, current_date, end_date)
    t3 = time.time()
    
    # create gamma vector for FIP0081
    update_day = current_date + timedelta(days=30)
    no_fip0081 = np.ones(forecast_length_days)
    gamma_smooth_1y = create_gamma_vector(update_day, forecast_length_days, current_date, ramp_len_days=int(365))

    # run simulation for the configured scenario, and for a pessimsitc and optimistic version of it
    gamma_configured = gamma_smooth_1y if st.session_state['options'] == 'FIP0081 (1Y Ramp)' else no_fip0081
    use_as_configured = st.session_state['options'] == 'CS->AS'
    scenarios = ['status-quo', 'configured']
    scenario_configs = {
        'status-quo': {
            'sector_duration_days': 360, 
            'lock_target': 0.3, 
            'gamma': no_fip0081, 
            'use_available_supply': False
        },
        'configured': {
            'sector_duration_days': sector_duration_days, 
            'lock_target': lock_target, 
            'gamma': gamma_configured,
            'use_available_supply': use_as_configured,
        },
    }
    scenario_results = {}
    for scenario in scenarios:
        scenario_scaler = 1.0
        rbp_val = rb_onboard_power_pib_day * scenario_scaler
        rr_val = max(0.0, min(1.0, renewal_rate_pct / 100. * scenario_scaler))
        fpr_val = max(0.0, min(1.0, fil_plus_rate_pct / 100. * scenario_scaler))

        rbp = jnp.ones(forecast_length_days) * rbp_val
        rr = jnp.ones(forecast_length_days) * rr_val
        fpr = jnp.ones(forecast_length_days) * fpr_val
        
        simulation_results = sim.run_sim(
            rbp, 
            rr, 
            fpr, 
            scenario_configs[scenario]['lock_target'], 
            start_date, 
            current_date, 
            forecast_length_days, 
            scenario_configs[scenario]['sector_duration_days'],
            offline_data,
            gamma=scenario_configs[scenario]['gamma'],
            gamma_weight_type=0,  # arithmetic weighting
            use_available_supply=scenario_configs[scenario]['use_available_supply'],
        ) 
        scenario_results[scenario] = simulation_results

    baseline = minting.compute_baseline_power_array(
        np.datetime64(start_date), np.datetime64(end_date), offline_data['init_baseline_eib'],
    )

    # plot
    plot_panel(scenario_results, baseline, start_date, current_date, end_date)
    t4 = time.time()
    
def main():
    st.set_page_config(
        page_title="Filecoin Economics Explorer",
        page_icon="🚀",  # TODO: can update this to the FIL logo
        layout="wide",
    )
    current_date = date.today() - timedelta(days=3)
    mo_start = max(current_date.month - 6 % 12, 1)
    start_date = date(current_date.year, mo_start, 1)
    forecast_length_days=365*3
    end_date = current_date + timedelta(days=forecast_length_days)
    forecast_kwargs = {
        'start_date': start_date,
        'current_date': current_date,
        'end_date': end_date,
        'forecast_length_days': forecast_length_days,
    }

    _, smoothed_last_historical_rbp, smoothed_last_historical_rr, smoothed_last_historical_fpr = get_offline_data(start_date, current_date, end_date)
    smoothed_last_historical_renewal_pct = int(smoothed_last_historical_rr * 100)
    smoothed_last_historical_fil_plus_pct = int(smoothed_last_historical_fpr * 100)
    
    with st.sidebar:
        st.title('Filecoin Economics Explorer')

        st.slider("Raw Byte Onboarding (PiB/day)", min_value=3., max_value=50., value=smoothed_last_historical_rbp, step=.1, format='%0.02f', key="rbp_slider",
                on_change=forecast_economy, kwargs=forecast_kwargs, disabled=False, label_visibility="visible")
        st.slider("Renewal Rate (Percentage)", min_value=10, max_value=99, value=smoothed_last_historical_renewal_pct, step=1, format='%d', key="rr_slider",
                on_change=forecast_economy, kwargs=forecast_kwargs, disabled=False, label_visibility="visible")
        st.slider("FIL+ Rate (Percentage)", min_value=10, max_value=99, value=smoothed_last_historical_fil_plus_pct, step=1, format='%d', key="fpr_slider",
                on_change=forecast_economy, kwargs=forecast_kwargs, disabled=False, label_visibility="visible")
        st.slider("Lock Target", min_value=0.05, max_value=0.5, value=0.3, step=0.01, format='%0.02f', key="lock_target_slider",
                on_change=forecast_economy, kwargs=forecast_kwargs, disabled=False, label_visibility="visible")
        st.slider("Sector Duration", min_value=180, max_value=720, value=360, step=30, format='%d', key="sector_duration_slider",
                  on_change=forecast_economy, kwargs=forecast_kwargs, disabled=False, label_visibility="visible")
        
        st.radio("FIP Modifications", options=['None', 'FIP0081 (1Y Ramp)', 'CS->AS'], index=0, key="options",
                 on_change=forecast_economy, kwargs=forecast_kwargs, disabled=False, label_visibility="visible")

        st.button("Forecast", on_click=forecast_economy, kwargs=forecast_kwargs, key="forecast_button")

    
    if "debug_string" in st.session_state:
        st.markdown(
            f'<div class="debug">{ st.session_state["debug_string"]}</div>',
            unsafe_allow_html=True,
        )
    components.html(
        d.js_code(),
        height=0,
        width=0,
    )

if __name__ == '__main__':
    main()