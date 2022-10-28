import altair as alt

""" Programmatic shortcuts for producing interactive plots using Altair
visualization library
"""


def hist_w_error_wildlife(
        field_x,
        brush,
        crossfilter,
        noisy_count="noisy_count",
        plus_error="plus_error",
        minus_error="minus_error",
        true_count="true_count",
        bar_color="#1167B1",
        display_true=False,
        label=False,
        projection=False,
):
    hist_color = bar_color
    opacity1 = 0.6
    curr_width = 25
    hist = alt.Chart().mark_bar(color=hist_color, opacity=opacity1, width=curr_width)

    x = alt.X(field=field_x, type="ordinal", sort=None)
    y = alt.Y(field=noisy_count, type="quantitative", aggregate="sum")

    base = (
        alt.Chart(width=550, height=350)
        .mark_bar()
        .add_selection(brush)
        .encode(
            x,
            color=alt.value("transparent"),
            tooltip=[
                alt.Tooltip("sum(" + noisy_count + "):Q", title="noisy count"),
                alt.Tooltip("sum(" + "noisy_count_prev" + "):Q", title="prev count"),
                alt.Tooltip("sum(" + "error" + "):Q", title="current error"),
                alt.Tooltip("sum(" + "error_prev" + "):Q", title="prev error"),
            ],
        )
        .transform_filter(crossfilter)
    )

    hist = hist.transform_filter(crossfilter).encode(x, y)
    error_bar = (
        alt.Chart()
        .transform_filter(crossfilter)
        .mark_rule(color="black")
        .encode(
            x=x,
            y=alt.Y(plus_error + ":Q", aggregate="sum", title=""),
            y2=alt.Y2(minus_error + ":Q", aggregate="sum", title=""),
        )
    )

    tick_up = (
        alt.Chart()
        .transform_filter(crossfilter)
        .mark_tick(color="black", width=20)
        .encode(x=x, y=alt.Y(plus_error + ":Q", aggregate="sum"))
    )

    tick_bottom = (
        alt.Chart()
        .transform_filter(crossfilter)
        .mark_tick(color="black", width=20)
        .encode(x=x, y=alt.Y(minus_error + ":Q", aggregate="sum"))
    )

    point = (
        alt.Chart()
        .transform_filter(crossfilter)
        .mark_point(color="black", opacity=opacity1)
        .encode(x=x, y=alt.Y(noisy_count + ":Q", aggregate="sum", title="Noisy Count"))
    )

    if display_true:
        true_mark = (
            alt.Chart()
            .transform_filter(crossfilter)
            .mark_point(color="red")
            .encode(x=x, y=alt.Y(true_count + ":Q", aggregate="sum"))
        )
        return alt.layer(base, hist, error_bar, tick_up, tick_bottom, true_mark)
    if projection:
        hist_color = "gray"
        opacity2 = 0.4
        prev_y = alt.Y(
            field=noisy_count + "_prev", type="quantitative", aggregate="sum"
        )

        hist_prev = alt.Chart().mark_bar(color=hist_color, opacity=opacity2)
        hist_prev = hist_prev.transform_filter(crossfilter).encode(x, prev_y)
        error_bar_prev = (
            alt.Chart()
            .transform_filter(crossfilter)
            .mark_rule(color="#3ddc65", strokeDash=[1, 1])
            .encode(
                x=x,
                y=alt.Y(plus_error + "_prev" + ":Q", aggregate="sum", title=""),
                y2=alt.Y2(minus_error + "_prev" + ":Q", aggregate="sum", title=""),
            )
        )

        tick_up_prev = (
            alt.Chart()
            .transform_filter(crossfilter)
            .mark_tick(color="#3ddc65", strokeDash=[1, 1], size=20)
            .encode(x=x, y=alt.Y(plus_error + "_prev" + ":Q", aggregate="sum"))
        )

        tick_bottom_prev = (
            alt.Chart()
            .transform_filter(crossfilter)
            .mark_tick(color="#3ddc65", strokeDash=[3, 1], size=20)
            .encode(x=x, y=alt.Y(minus_error + "_prev" + ":Q", aggregate="sum"))
        )

        return alt.layer(
            base,
            hist_prev,
            hist,
            error_bar,
            tick_up,
            tick_bottom,
            point,
            error_bar_prev,
            tick_up_prev,
            tick_bottom_prev,
        )
    if label:
        hist_color = "gray"
        opacity2 = 0.4
        prev_y = alt.Y(
            field=noisy_count + "_prev", type="quantitative", aggregate="sum"
        )

        hist_prev = alt.Chart().mark_bar(color=hist_color, opacity=opacity2)
        hist_prev = hist_prev.transform_filter(crossfilter).encode(x, prev_y)

        text = (
            tick_up.mark_text(
                align="center", baseline="top", dy=-25, lineBreak="x", fontSize=9
            )
            .transform_joinaggregate(cerror="sum(error):Q", groupby=[field_x])
            .transform_joinaggregate(perror="sum(error_prev):Q", groupby=[field_x])
            .transform_calculate(
                label='"curr: " + datum.cerror + "x prev: " +datum.perror'
            )
            .encode(text=alt.Text("label:N"))
        )

        return alt.layer(
            base, hist_prev, hist, error_bar, tick_up, tick_bottom, point, text
        )

    else:
        return alt.layer(base, hist, error_bar, tick_up, tick_bottom, point)


def hist_w_error(
        field_x,
        brush,
        crossfilter,
        noisy_count="noisy_count",
        plus_error="plus_error",
        minus_error="minus_error",
        true_count="true_count",
        display_true=False,
        label=False,
        projection=False,
):
    hist_color = "#1167B1"
    opacity1 = 0.6
    curr_width = 30
    hist = alt.Chart().mark_bar(color=hist_color, opacity=opacity1, width=curr_width)

    x = alt.X(field=field_x, type="ordinal", sort=None)
    y = alt.Y(field=noisy_count, type="quantitative", aggregate="sum")

    base = (
        alt.Chart(width=500, height=350)
        .mark_bar()
        .add_selection(brush)
        .encode(
            x,
            color=alt.value("transparent"),
            tooltip=[
                alt.Tooltip("sum(" + noisy_count + "):Q", title="noisy count"),
                # alt.Tooltip('sum(' + plus_error + '):Q', title='upper range'),
                # alt.Tooltip('sum(' + minus_error + '):Q', title='lower range'),
                alt.Tooltip("sum(" + "error" + "):Q", title="current error"),
                alt.Tooltip("sum(" + "error_prev" + "):Q", title="prev error"),
            ],
        )
        .transform_filter(crossfilter)
    )

    hist = hist.transform_filter(crossfilter).encode(x, y)
    error_bar = (
        alt.Chart()
        .transform_filter(crossfilter)
        .mark_rule(color="black")
        .encode(
            x=x,
            y=alt.Y(plus_error + ":Q", aggregate="sum", title=""),
            y2=alt.Y2(minus_error + ":Q", aggregate="sum", title=""),
        )
    )

    tick_up = (
        alt.Chart()
        .transform_filter(crossfilter)
        .mark_tick(color="black", width=20)
        .encode(x=x, y=alt.Y(plus_error + ":Q", aggregate="sum"))
    )

    tick_bottom = (
        alt.Chart()
        .transform_filter(crossfilter)
        .mark_tick(color="black", width=20)
        .encode(x=x, y=alt.Y(minus_error + ":Q", aggregate="sum"))
    )

    point = (
        alt.Chart()
        .transform_filter(crossfilter)
        .mark_point(color="black", opacity=opacity1)
        .encode(x=x, y=alt.Y(noisy_count + ":Q", aggregate="sum", title="Noisy Count"))
    )

    if display_true:
        true_mark = (
            alt.Chart()
            .transform_filter(crossfilter)
            .mark_point(color="red")
            .encode(x=x, y=alt.Y(true_count + ":Q", aggregate="sum"))
        )
        return alt.layer(base, hist, error_bar, tick_up, tick_bottom, true_mark)
    if projection:
        hist_color = "gray"
        opacity2 = 0.4
        prev_y = alt.Y(
            field=noisy_count + "_prev", type="quantitative", aggregate="sum"
        )

        hist_prev = alt.Chart().mark_bar(color=hist_color, opacity=opacity2)
        hist_prev = hist_prev.transform_filter(crossfilter).encode(x, prev_y)
        error_bar_prev = (
            alt.Chart()
            .transform_filter(crossfilter)
            .mark_rule(color="#3ddc65", strokeDash=[1, 1])
            .encode(
                x=x,
                y=alt.Y(plus_error + "_prev" + ":Q", aggregate="sum", title=""),
                y2=alt.Y2(minus_error + "_prev" + ":Q", aggregate="sum", title=""),
            )
        )

        tick_up_prev = (
            alt.Chart()
            .transform_filter(crossfilter)
            .mark_tick(color="#3ddc65", strokeDash=[1, 1], size=30)
            .encode(x=x, y=alt.Y(plus_error + "_prev" + ":Q", aggregate="sum"))
        )

        tick_bottom_prev = (
            alt.Chart()
            .transform_filter(crossfilter)
            .mark_tick(color="#3ddc65", strokeDash=[3, 1], size=30)
            .encode(x=x, y=alt.Y(minus_error + "_prev" + ":Q", aggregate="sum"))
        )

        return alt.layer(
            base,
            hist_prev,
            hist,
            error_bar,
            tick_up,
            tick_bottom,
            point,
            error_bar_prev,
            tick_up_prev,
            tick_bottom_prev,
        )
    if label:
        hist_color = "gray"
        opacity2 = 0.4
        prev_y = alt.Y(
            field=noisy_count + "_prev", type="quantitative", aggregate="sum"
        )

        hist_prev = alt.Chart().mark_bar(color=hist_color, opacity=opacity2)
        hist_prev = hist_prev.transform_filter(crossfilter).encode(x, prev_y)

        text = (
            tick_up.mark_text(
                align="center", baseline="top", dy=-25, lineBreak="x", fontSize=9
            )
            .transform_joinaggregate(cerror="sum(error):Q", groupby=[field_x])
            .transform_joinaggregate(perror="sum(error_prev):Q", groupby=[field_x])
            .transform_calculate(
                label='"curr: " + datum.cerror + "x prev: " +datum.perror'
            )
            .encode(text=alt.Text("label:N"))
        )

        return alt.layer(
            base, hist_prev, hist, error_bar, tick_up, tick_bottom, point, text
        )
    else:
        return alt.layer(base, hist, error_bar, tick_up, tick_bottom, point)


def linked_hist(
        field_x1,
        field_x2,
        data,
        count_fieldname="noisy_count",
        display_true=False,
        display_error=False,
        history=False,
):
    """Produces a linked histogram over ``field_x1`` and ``field_x2`` using
    information from ``data`` by first creating a linked histogram from x1 to
    x2, then creating one from x2 to x1 and finally concatenating that
    horizontally.
    Args:
            field_x1 (str): Column name corresponding to chart to displayed on the
                    left.
            field_x2 (str): Column name corresponding to chart to be dispayed
                    on the right.
            data (DataFrame): DataFrame containing visualization specification.
            count_fieldname (str, optional): The name of the column which contains
                    the counts to displayed. Default is set to ``noisy_count``.
            display_true: ``True`` if displaying true counts, ``False`` otherwise.
                    Meant for demos since displaying the true values violates differential
                    privacy.
            display_error: ``True`` if displaying error bars, ``False`` otherwise.
    """

    def interactive_hist(field_x, field_y, brush, crossfilter):
        hist_color = "#1167B1"
        opacity1 = 0.7
        hist = alt.Chart().mark_bar(color=hist_color, opacity=opacity1)
        x = alt.X(field=field_x, type="ordinal", sort=None)
        y = alt.Y(field=field_y, type="quantitative", aggregate="sum")

        base = (
            alt.Chart(width=500, height=350)
            .mark_bar()
            .add_selection(brush)
            .encode(
                x,
                color=alt.value("transparent"),
                tooltip=[
                    alt.Tooltip("sum(noisy_count):Q", title="noisy count"),
                    alt.Tooltip("sum(error):Q", title="error"),
                ],
            )
            .transform_filter(crossfilter)
        )

        hist = hist.transform_filter(crossfilter).encode(x, y)
        error_bar = (
            alt.Chart()
            .transform_filter(crossfilter)
            .mark_rule()
            .encode(
                x=x,
                y=alt.Y("plus_error:Q", aggregate="sum", title=""),
                y2=alt.Y2("minus_error:Q", aggregate="sum", title=""),
            )
        )

        tick_up = (
            alt.Chart()
            .transform_filter(crossfilter)
            .mark_tick(color="black", width=20)
            .encode(x=x, y=alt.Y("plus_error:Q", aggregate="sum"))
        )

        tick_bottom = (
            alt.Chart()
            .transform_filter(crossfilter)
            .mark_tick(color="black", width=20)
            .encode(x=x, y=alt.Y("minus_error:Q", aggregate="sum"))
        )

        point = (
            alt.Chart()
            .transform_filter(crossfilter)
            .mark_point(color="black")
            .encode(x=x, y=alt.Y("noisy_count:Q", aggregate="sum", title="Noisy Count"))
        )

        if display_true:
            true_mark = (
                alt.Chart()
                .transform_filter(crossfilter)
                .mark_point(color="red")
                .encode(x=x, y=alt.Y("true_count:Q", aggregate="sum"))
            )
            return alt.layer(base, hist, error_bar, tick_up, tick_bottom, true_mark)
        if history:
            pass
        else:
            return alt.layer(base, hist, tick_up, tick_bottom, point, error_bar)

    brush_x1 = alt.selection(type="interval", encodings=["x"])
    brush_x2 = alt.selection(type="interval", encodings=["x"])
    return (
        alt.hconcat(
            interactive_hist(field_x1, count_fieldname, brush_x1, brush_x2),
            interactive_hist(field_x2, count_fieldname, brush_x2, brush_x1),
            data=data,
        )
        .configure_axis(
            labelFontSize=11,
            titleFontSize=15,
        )
        .configure_axisBottom(labelAngle=60)
    )


def linked_hist_test(field_x1, field_x2, data, projection=False, label=False):
    brush_x1 = alt.selection(type="interval", encodings=["x"])
    brush_x2 = alt.selection(type="interval", encodings=["x"])
    return (
        alt.hconcat(
            hist_w_error_wildlife(
                field_x1,
                brush=brush_x1,
                crossfilter=brush_x2,
                projection=projection,
                display_true=False,
                label=label,
            ),
            hist_w_error_wildlife(
                field_x2,
                brush=brush_x2,
                crossfilter=brush_x1,
                projection=projection,
                display_true=False,
                label=label,
            ),
            data=data,
        )
        .configure_axis(
            labelFontSize=11,
            titleFontSize=15,
        )
        .configure_axisBottom(labelAngle=60)
    )


def categorical_plot(categorical_field, field_y):
    """Produces a bar chart over a categorical variable with quantitative
            values.
    Args:
            categorical_field (str): Name of categorical field.
            field_y (str): Name of field where quantitative value of
                    ``categorical_field`` is stored.
    """
    selection = alt.selection_multi()
    x = alt.X(field=categorical_field, type="categorical", sort=None)
    y = alt.Y(field=field_y, type="quantitative", aggregate="sum")

    base = (
        alt.Chart()
        .mark_bar()
        .add_selection(selection)
        .encode(x, color=alt.value("transparent"))
    )
