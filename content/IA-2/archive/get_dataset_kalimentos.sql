set
    time zone 'America/Sao_Paulo';

select
    ped.id as pedido_id,
    ped.total as pedido_valor_total,
    ped.data as pedido_data,
    ped.discount_type as pedido_tipo_desconto,
    ped.discount_value as pedido_valor_desconto,
    ped.subtotal as pedido_subtotal,
    ped.user_id as pedido_vendedor,
    ped.payment_method as pedido_metodo_pagamento,
    iped.produto_nome as item_pedido_nome_produto,
    iped.quantidade as item_pedido_quantidade,
    iped.preco_unitario as item_pedido_preco_unitario,
    iped.user_id as item_pedido_user_id,
    ev.name as evento_nome,
from
    itens_pedido as iped
    left join pedidos as ped on iped.pedido_id = ped.id
    left join finance_events as ev on ped.event_id = ev.id
where
    ped.event_id is not null
    and ped.cliente_id is null;