<script>
    import { sparse_color_map_css } from "./shared_utils/colors";
    import { default as ArrayImage } from "./components/ArrayImage.svelte";
    import LockableValueToggle from "./components/LockableValueToggle.svelte";

    // Pass in values for these
    export let tokens;
    export let attention; // the original/input. the one to use/display should be attention_show (below)
    export let pos_logits;
    export let neg_logits;
    export let head_labels;
    export let show_tokens = true;
    // export let blank_color = "#FFF";


    export let focus_token_lock = { value: undefined, mode: "soft" };
    $: focus_token = focus_token_lock.value;

    export let focus_head_lock = { value: undefined, mode: "soft" };
    $: focus_head = focus_head_lock.value;

    export let hover_token_is_target = false;

    // For a display element that isn't slow, can abstract over the switching between the original vs. info-weighted
    // by just using attention_show and letting svelte handle it.
    // However, as a heads-up for later in this fle, it can be too slow to re-draw / re-compute some aspects.
    // In the case of the ArrayImage components, we create both up front, and then display only the active one.
    $: attention_show = attention
    $: window.attention_show = attention_show;

    export let _show_logits = false;
    export let _show_neg_logits = false;

    function range(n) {
        return [...Array(n).keys()];
    }

    function reduce_Y(arr){
        if (arr === undefined) {
            return undefined
        }
        var arr_ = [];
        for (var x=0; x < arr.shape[0]; x++){
            arr_.push([])
            for (var c=0; c < arr.shape[2]; c++){
                var temp = 0;
                for (var y=0; y < arr.shape[1]; y++){
                    temp = Math.max(temp, arr.pick(x,y,c));
                }
                arr_[x].push(temp);
            }
        }
        return arr_;
    }
    function reduce_X(arr){
        if (arr === undefined) {
            return undefined
        }
        var arr_ = [];
        for (var y=0; y < arr.shape[0]; y++){
            arr_.push([])
            for (var c=0; c < arr.shape[2]; c++){
                var temp = 0;
                for (var x=0; x < arr.shape[1]; x++){
                    temp = Math.max(temp, arr.pick(x,y,c));
                }
                arr_[y].push(temp);
            }
        }
        return arr_;
    }


    // Cache both versions, to do less computation on them when switching
    $: attention_reduce_dst = reduce_Y(attention);
    $: attention_reduce_src = reduce_X(attention);
    
    $: pos_logits_reduce_dst = reduce_Y(pos_logits);
    $: pos_logits_reduce_src = reduce_X(pos_logits);
    
    $: neg_logits_reduce_dst = reduce_Y(neg_logits);
    $: neg_logits_reduce_src = reduce_X(neg_logits);


    $: N_heads = attention.shape[2];
    $: colors = range(N_heads).map((i) =>
        sparse_color_map_css(
            range(N_heads).map((x) => 1),
            undefined,
            i
        )
    );
    $: head_labels_ = head_labels != undefined ? head_labels : range(N_heads);

    function get_color(array, x, y, isolate_channel = undefined) {
        if (x < y) {
            return "#FFF";
        }
        var v = array.pick(x, y, null);
        return sparse_color_map_css(v, undefined, isolate_channel);
    }

    function head_intensity(
        head_i,
        focus_token_value,
        hover_token_is_target,
        src_values,
        dst_values,
    ) {
        if (focus_token_value == undefined) {
            var v = 1.0
        } else {
            var reduced_array = hover_token_is_target? src_values : dst_values;
            var v = Math.max(0, Math.min(1, reduced_array[focus_token_value][head_i]));
        }
        return ""+v;
    }

   function token_color(
        array,
        focus_token_value,
        tok_i,
        isolate_channel = undefined,
        hover_token_is_target,
        src_values,
        dst_values,
    ) {
        if (focus_token_value == undefined) {
            var reduced_array = hover_token_is_target? src_values : dst_values;
            return sparse_color_map_css(reduced_array[tok_i], undefined, isolate_channel);
            //return blank_color;
        }

        let tok_from = undefined;
        let tok_to = undefined;
        if (hover_token_is_target) {
            tok_from = tok_i;
            tok_to = focus_token_value;
        } else {
            tok_from = focus_token_value;
            tok_to = tok_i;
        }
        return get_color(array, tok_from, tok_to, isolate_channel);
    }

    export let all_attn_token_colors;
    export let all_pos_logits_token_colors;
    export let all_neg_logits_token_colors;

    $: all_attn_token_colors = range(tokens.length).map(
        i => token_color(
                        attention_show,
                        focus_token,
                        i,
                        focus_head,
                        hover_token_is_target,
                        attention_reduce_src,
                        attention_reduce_dst
                    ));
    $: all_pos_logits_token_colors = range(tokens.length).map(
        i => token_color(
                        pos_logits,
                        focus_token,
                        i,
                        focus_head,
                        hover_token_is_target,
                        pos_logits_reduce_src,
                        pos_logits_reduce_dst
                    ));
    $: all_neg_logits_token_colors = range(tokens.length).map(
        i => token_color(
                        neg_logits,
                        focus_token,
                        i,
                        focus_head,
                        hover_token_is_target,
                        neg_logits_reduce_src,
                        neg_logits_reduce_dst
                    ));
</script>

<div class="attn-container">
    <div class="figcaption" style="grid-column: big-attn;">
        Attention Pattern
        {#if focus_head != undefined}
            ({head_labels_[focus_head]})
        {/if}
    </div>
    {#if _show_logits}
        {#if _show_neg_logits}
            <div style="grid-column: big-attn; grid-row: main;">
                <ArrayImage
                array={neg_logits}
                width="200"
                height="200"
                {focus_token}
                isolate_channel={focus_head}
                />
            </div>
        {:else}
        
            <div style="grid-column: big-attn; grid-row: main;">
                <ArrayImage
                array={pos_logits}
                width="200"
                height="200"
                {focus_token}
                isolate_channel={focus_head}
                />
            </div>
        
        {/if}
    {:else}
        <div style="grid-column: big-attn; grid-row: main;">
        <ArrayImage
            array={attention}
            width="200"
            height="200"
            {focus_token}
            isolate_channel={focus_head}
        />
        </div>
    {/if}
    {#if N_heads > 1}
        <div class="figcaption" style="grid-column: heads;">
            Attention Heads (hover to focus, click to lock)
        </div>
        <div class="heads">
            {#each range(attention_show.shape[2]) as head_i}
                <LockableValueToggle
                    bind:lock={focus_head_lock}
                    set_value={head_i}
                >
                    {#if _show_logits}
                        {#if _show_neg_logits}
                            <div
                                class="head-icon"
                                style="opacity: {focus_head != undefined &&
                                focus_head != head_i
                                    ? '0.2'
                                    : head_intensity(head_i, focus_token, hover_token_is_target, neg_logits_reduce_src, neg_logits_reduce_dst)};"
                            >
                                <ArrayImage
                                    array={neg_logits}
                                    width="60"
                                    height="60"
                                    isolate_channel={head_i}
                                />
                                <div
                                    class="head-label"
                                    style="background: {colors[head_i]}"
                                >
                                    {@html head_labels_[head_i] != undefined
                                        ? head_labels_[head_i]
                                        : "&nbsp"}
                                </div>
                            </div>
                        {:else}
                            <div
                                class="head-icon"
                                style="opacity: {focus_head != undefined &&
                                focus_head != head_i
                                    ? '0.2'
                                    : head_intensity(head_i, focus_token, hover_token_is_target, pos_logits_reduce_src, pos_logits_reduce_dst)};"
                            >
                                <ArrayImage
                                    array={pos_logits}
                                    width="60"
                                    height="60"
                                    isolate_channel={head_i}
                                />
                                <div
                                    class="head-label"
                                    style="background: {colors[head_i]}"
                                >
                                    {@html head_labels_[head_i] != undefined
                                        ? head_labels_[head_i]
                                        : "&nbsp"}
                                </div>
                            </div>
                        {/if}
                    {:else}
                        <div
                            class="head-icon"
                            style="opacity: {focus_head != undefined &&
                            focus_head != head_i
                                ? '0.2'
                                : head_intensity(head_i, focus_token, hover_token_is_target, attention_reduce_src, attention_reduce_dst)};"
                        >
                            <ArrayImage
                                array={attention}
                                width="60"
                                height="60"
                                isolate_channel={head_i}
                            />
                            <div
                                class="head-label"
                                style="background: {colors[head_i]}"
                            >
                                {@html head_labels_[head_i] != undefined
                                    ? head_labels_[head_i]
                                    : "&nbsp"}
                            </div>
                        </div>
                    {/if}
                    </LockableValueToggle>
            {/each}
        </div>
    {/if}
</div>

{#if show_tokens}
<div class="tokens-container">
    <div class="figcaption" style="grid-column: left;">
        Tokens (hover to focus, click to lock)
    </div>
    <div class="tokens">
        {#each tokens as tok, tok_i}
            <!-- n.b. it's important to not have any whitespace inside
                the LockableValueToggle tag below, which is why we
                format it like we do. See the comment inside
                LockableValueToggle.html for more detail.
              -->
            <LockableValueToggle bind:lock={focus_token_lock}
                                 set_value={tok_i}
                                 style='display: inline'><span
                    class="token {tok_i == focus_token ? 'selected' : ''}"
                    style="background: {_show_logits ? (_show_neg_logits ? all_neg_logits_token_colors[tok_i] : all_pos_logits_token_colors[tok_i]) : all_attn_token_colors[tok_i]};"
                >{tok}</span></LockableValueToggle>
        {/each}
    </div>
    <div class="toggle"><nobr>
        <input class="hover-mode" type="checkbox" bind:checked={hover_token_is_target}/>
        <span class="hover-mode-text"
              style="white-space: nowrap;"
              on:click={() => (hover_token_is_target = (hover_token_is_target ^ true))}>
            Selected is
            <b>{hover_token_is_target ? "target" : "source"}</b></span>
        <input class="hover-mode" type="checkbox" bind:checked={_show_logits}/>
        <span class="hover-mode-text"
              style="white-space: nowrap;"
              on:click={() => (_show_logits = (_show_logits ^ true))}>
            Displaying
            <b>{_show_logits ? "Logit attribution" : "Attention"}</b></span>
        {#if _show_logits}
            <input class="hover-mode" type="checkbox" bind:checked={_show_neg_logits}/>
            <span class="hover-mode-text"
                style="white-space: nowrap;"
                on:click={() => (_show_neg_logits = (_show_neg_logits ^ true))}>
                Showing
                <b>{_show_neg_logits ? "negative contribution" : "positive contribution"}</b></span>
        {/if}
    </nobr></div>
</div>
{/if}

<style>
    .attn-container {
        display: grid;
        grid-template-rows: [title] min-content [main] min-content;
        grid-template-columns: [big-attn] min-content [heads] minmax(min-content, 624px);
        gap: 12px;
    }
    .figcaption {
        color: #888;
        grid-row: title;
        white-space: nowrap;
    }
    .tokens-container {
        display: grid;
        grid-template-rows: [title] min-content [main] min-content;
        grid-template-columns: [left] min-content [right] minmax(min-content, 800px) [end];
        gap: 12px;
        margin-top: 24px;
    }
    .tokens {
        grid-row: main;
        grid-column-start: left;
        grid-column-end: end;
        cursor: pointer;
        height: min-content;
        line-height: 110%;
    }
    .tokens .token {
        white-space: pre-wrap;
    }
    .tokens .selected {
        border: 1px solid #999;
        z-index: 10;
    }
    .tokens .token:not(.selected) {
        z-index: 0;
        padding: 1px;
    }
    .hover-mode, .hover-mode-text {
        color: #888;
        grid-row: title;
        grid-column: settings;
        cursor: pointer;
    }
    .hover-mode-text {
        margin-right: 8px;
    }
    .heads {
        grid-column: heads;
        grid-row: main;
        display: flex;
        flex-direction: row;
        flex-wrap: wrap;
        gap: 6px;
        height: min-content;
    }
    .heads .head-icon {
        position: relative;
        width: 62px;
        height: 62px;
    }
    .heads .head-icon > * {
        position: absolute;
        right: 0px;
        top: 0px;
    }
    .heads .head-icon .head-label {
        background: #333;
        color: #eee;
        font-size: 65%;
        padding: 1px;
        border-bottom-left-radius: 2px;
        padding-left: 4px;
        padding-right: 2px;
        min-width: 14px;
        opacity: 0.75;
    }
</style>
