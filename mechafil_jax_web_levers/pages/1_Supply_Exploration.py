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

import pystarboard.data as psb
import pystarboard.data_spacescope as psb_sp

import mechafil_jax.data as data
import mechafil_jax.sim as sim
import mechafil_jax.constants as C
import mechafil_jax.minting as minting
import mechafil_jax.date_utils as du

import scenario_generator.utils as u

def create_gamma_trajectory(current_date, forecast_length_days, fip81_activation_date, ramp_len_days=365):
    gamma_target = 0.7
    days_since_activation = (current_date - fip81_activation_date).days
    gamma_slope = (1.0 - gamma_target) / ramp_len_days
    current_gamma = 1.0 - gamma_slope * days_since_activation
    print(f'current_gamma: {current_gamma}')
    remaining_days = ramp_len_days - days_since_activation
    v1 = np.linspace(current_gamma, gamma_target, remaining_days)
    v2 = np.ones(forecast_length_days - remaining_days) * gamma_target
    gamma_trajectory = np.concatenate([v1, v2])

    return gamma_trajectory

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

    # get historical pledge/locked/circ-supply, as these have a discrepancy with MechaFIL.
    # dss = psb_sp.SpacescopeDataConnection(PUBLIC_AUTH_TOKEN)
    # sector_economics_df = dss.get_sector_economics_stats(start_date, current_date)

    
    # hist_df = psb.get_historical_network_stats(start_date-timedelta(days=180), current_date, current_date)
    # print(hist_df)

    # return offline_data, smoothed_last_historical_rbp, smoothed_last_historical_rr, smoothed_last_historical_fpr, sector_economics_df
    return offline_data, smoothed_last_historical_rbp, smoothed_last_historical_rr, smoothed_last_historical_fpr

def plot_panel(scenario_results, baseline, start_date, current_date, end_date):
    # convert results dictionary into a dataframe so that we can use altair to make nice plots
    status_quo_results = scenario_results['status-quo']
    configured_results = scenario_results['configured']

    col1, col2, col3, col4 = st.columns(4)

    date_vec = pd.to_datetime(du.get_t(current_date, end_date=end_date))
    
    power_dff = pd.DataFrame()
    power_dff['RBP'] = status_quo_results['rb_total_power_eib'][-len(date_vec):]
    power_dff['QAP'] = status_quo_results['qa_total_power_eib'][-len(date_vec):]
    power_dff['Baseline'] = baseline[-len(date_vec):]
    power_dff['date'] = date_vec

    minting_dff = pd.DataFrame()
    minting_dff['StatusQuo'] = status_quo_results['day_network_reward'][-len(date_vec):]
    minting_dff['Configured'] = configured_results['day_network_reward'][-len(date_vec):]
    minting_dff['date'] = date_vec

    cs_dff = pd.DataFrame()
    cs_dff['StatusQuo'] = status_quo_results['circ_supply'][-len(date_vec):] / 1e6
    cs_dff['Configured'] = configured_results['circ_supply'][-len(date_vec):] / 1e6
    cs_dff['date'] = date_vec

    cs_delta_dff = pd.DataFrame()
    cs_sq = np.diff(np.asarray(status_quo_results['circ_supply']) / 1e6)
    cs_conf = np.diff(np.asarray(configured_results['circ_supply']) / 1e6)
    cs_delta_dff['StatusQuo'] = cs_sq[-len(date_vec):]
    cs_delta_dff['Configured'] = cs_conf[-len(date_vec):]
    cs_delta_dff['date'] = date_vec

    as_dff = pd.DataFrame()
    as_dff['StatusQuo'] = status_quo_results['available_supply'][-len(date_vec):] / 1e6
    as_dff['Configured'] = configured_results['available_supply'][-len(date_vec):] / 1e6
    as_dff['date'] = date_vec
    as_dff = as_dff[1:]

    locked_dff = pd.DataFrame()
    locked_dff['StatusQuo'] = (status_quo_results['network_locked'][-len(date_vec):] / 1e6)
    locked_dff['Configured'] = (configured_results['network_locked'][-len(date_vec):] / 1e6)
    locked_dff['StatusQuo'] = locked_dff['StatusQuo'].rolling(window=180, min_periods=1).mean()
    locked_dff['Configured'] = locked_dff['Configured'].rolling(window=180, min_periods=1).mean()
    locked_dff['date'] = date_vec

    pledge_dff = pd.DataFrame()
    pledge_dff['StatusQuo'] = status_quo_results['day_pledge_per_QAP'][-len(date_vec):]
    pledge_dff['Configured'] = configured_results['day_pledge_per_QAP'][-len(date_vec):]
    pledge_dff['date'] = date_vec

    start_cur_day_delta = (current_date - start_date).days
    roi_dff = pd.DataFrame()
    roi_dff['StatusQuo_noFees'] = status_quo_results['1y_sector_roi'][start_cur_day_delta:] * 100
    roi_dff['StatusQuo [-FIP100]'] = status_quo_results['roi_with_fees_before_fip100'][start_cur_day_delta:] * 100
    roi_dff['StatusQuo [+FIP100]'] = status_quo_results['roi_with_fees_after_fip100'][start_cur_day_delta:] * 100
    roi_dff['Configured_noFees'] = configured_results['1y_sector_roi'][start_cur_day_delta:] * 100
    roi_dff['Configured [-FIP100]'] = configured_results['roi_with_fees_before_fip100'][start_cur_day_delta:] * 100
    roi_dff['Configured [+FIP100]'] = configured_results['roi_with_fees_after_fip100'][start_cur_day_delta:] * 100
    roi_dff['date'] = pd.to_datetime(du.get_t(current_date, forecast_length=roi_dff.shape[0]))

    # rps_dff = pd.DataFrame()
    # rps_dff['StatusQuo'] = status_quo_results['1y_return_per_sector'][start_cur_day_delta:] * 100
    # rps_dff['Configured'] = configured_results['1y_return_per_sector'][start_cur_day_delta:] * 100
    # rps_dff['date'] = pd.to_datetime(du.get_t(current_date, forecast_length=roi_dff.shape[0]))

    # drps_dff = pd.DataFrame()
    # drps_dff['StatusQuo'] = status_quo_results['day_rewards_per_sector'][-len(date_vec):] * 100
    # drps_dff['Configured'] = configured_results['day_rewards_per_sector'][-len(date_vec):] * 100
    # drps_dff['date'] = date_vec

    # dnr_dff = pd.DataFrame()
    # dnr_dff['StatusQuo'] = status_quo_results['day_network_reward'][-len(date_vec):] * 100
    # dnr_dff['Configured'] = configured_results['day_network_reward'][-len(date_vec):] * 100
    # dnr_dff['date'] = date_vec

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
                         value_vars=[
                            #  "StatusQuo_noFees", 
                             "StatusQuo [-FIP100]", 
                             "StatusQuo [+FIP100]", 
                            #  "Configured_noFees", 
                             "Configured [-FIP100]", 
                             "Configured [+FIP100]"
                         ], 
                         var_name='Scenario', 
                         value_name='%')
        roi = (
            alt.Chart(roi_df)
            .mark_line()
            .encode(x=alt.X("date", title="", axis=alt.Axis(format="%b %Y", labelAngle=-45, tickCount=5)), 
                    y=alt.Y("%"), 
                    color=alt.Color('Scenario', 
                                  scale=alt.Scale(
                                      domain=[
                                          'StatusQuo [-FIP100]', 
                                          'StatusQuo [+FIP100]', 
                                          'Configured [-FIP100]', 
                                          'Configured [+FIP100]'
                                        ],
                                        range=[
                                            '#4292c6', 
                                            '#6baed6', 
                                            '#ef6548', 
                                            '#fc9272'
                                        ],
                                  ),
                                  legend=alt.Legend(
                                      orient="left",
                                      title=None,
                                      direction="vertical",
                                      offset=0,
                                      symbolSize=100,
                                      labelFontSize=10,
                                      titleFontSize=10,
                                      padding=5
                                  )))
            .properties(title="1Y Sector FoFR")
        )
        final_roi = (roi + today_line).configure_title(fontSize=14, anchor='middle').configure_legend(
            labelColor='black',
            titleColor='black',
            strokeColor='gray',
            strokeWidth=1,
            padding=5,
            cornerRadius=5
        )
        st.altair_chart(final_roi.interactive(), use_container_width=True)

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
    #offline_data, _, _, _, sector_economics_df = get_offline_data(start_date, current_date, end_date)
    offline_data, _, _, _ = get_offline_data(start_date, current_date, end_date)
    t3 = time.time()
    # print(sector_economics_df[['date', 'sector_initial_pledge_32gib']])
    
    # create gamma vector for FIP0081
    fip81_activation_date = date(2024, 11, 21)
    gamma_smooth_1y = create_gamma_trajectory(current_date, forecast_length_days, fip81_activation_date, ramp_len_days=365)

    # run simulation for the configured scenario, and for a pessimsitc and optimistic version of it
    use_as_configured = st.session_state['options'] == 'CS->AS'
    scenarios = ['status-quo', 'configured']
    scenario_configs = {
        'status-quo': {
            'sector_duration_days': sector_duration_days, 
            'lock_target': 0.3, 
            'gamma': gamma_smooth_1y, 
            'use_available_supply': False
        },
        'configured': {
            'sector_duration_days': sector_duration_days, 
            'lock_target': lock_target, 
            'gamma': gamma_smooth_1y,
            'use_available_supply': use_as_configured,
        },
    }
    scenario_results = {}
    for scenario in scenarios:
        rbp_val = rb_onboard_power_pib_day
        rr_val = max(0.0, min(1.0, renewal_rate_pct / 100.))
        fpr_val = max(0.0, min(1.0, fil_plus_rate_pct / 100.))

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

        # compute before/after fees
        if rbp_val < 2.5:
            prefip_multiplier = 0.5
            fee_regime_scaler = 1
            F_postfip = 1
        else:
            prefip_multiplier = 0.95
            fee_regime_scaler = 2
            F_postfip = 0.0001
        before_fip_gasfees_per_sector = (offline_data['daily_burnt_fil'] * prefip_multiplier * fee_regime_scaler) / \
                                        ((simulation_results['day_onboarded_power_QAP_PIB'] + simulation_results['day_renewed_power_QAP_PIB']) * 1.0/C.PIB_PER_SECTOR)
        after_fip_gasfees_per_sector = before_fip_gasfees_per_sector * F_postfip
        delta_burn = jnp.clip(after_fip_gasfees_per_sector - before_fip_gasfees_per_sector, 0, 1e-10)
        simulation_results['circ_supply'] = simulation_results['circ_supply'] - delta_burn
        simulation_results['available_supply'] = simulation_results['available_supply'] - delta_burn
        simulation_results['before_fip_gasfees_per_sector'] = before_fip_gasfees_per_sector
        simulation_results['after_fip_gasfees_per_sector'] = after_fip_gasfees_per_sector
        fee_frac = 3e-12
        fee_key = 'circ_supply' if scenario=='status-quo' else 'available_supply'
        cs_fee = simulation_results[fee_key][1:] * fee_frac
        
        rps = simulation_results['1y_return_per_sector']
        len_rps = len(rps)
        rps_with_fees_before_fip100 = rps - before_fip_gasfees_per_sector[:len_rps]
        rps_with_fees_after_fip100 = rps - after_fip_gasfees_per_sector[:len_rps] - cs_fee[:len_rps]
        simulation_results['rps_with_fees_before_fip100'] = rps_with_fees_before_fip100
        simulation_results['rps_with_fees_after_fip100'] = rps_with_fees_after_fip100
        dppq = simulation_results['day_pledge_per_QAP']
        simulation_results['roi_with_fees_before_fip100'] = rps_with_fees_before_fip100/dppq[:len_rps]
        simulation_results['roi_with_fees_after_fip100'] = rps_with_fees_after_fip100/dppq[:len_rps]

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
    # mo_start = max(current_date.month - 6 % 12, 1)
    mo_start = current_date.month
    start_date = date(current_date.year, mo_start, 1)
    forecast_length_days=365*3
    end_date = current_date + timedelta(days=forecast_length_days)
    forecast_kwargs = {
        'start_date': start_date,
        'current_date': current_date,
        'end_date': end_date,
        'forecast_length_days': forecast_length_days,
    }

    # _, smoothed_last_historical_rbp, smoothed_last_historical_rr, smoothed_last_historical_fpr, _ = get_offline_data(start_date, current_date, end_date)
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
        
        st.radio("FIP Modifications", options=['Baseline (++ FIP0081)', 'CS->AS'], index=0, key="options",
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