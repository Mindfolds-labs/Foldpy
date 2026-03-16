# FoldPy como camada de inferência espacial aproximada (2D → pseudo-3D)

## Hipótese
O FoldPy pode transformar uma imagem 2D em pistas estruturais aproximadas de ocupação espacial (sem reconstrução 3D métrica), úteis como canais extras para CNN.

## Pipeline proposto
1. **Entrada 2D** (`ImageGrid`, escala de cinza em MNIST).
2. **Extração de bordas** (`edge_map`).
3. **Separação estrutural**:
   - `object_mask` via limiar simples (`mask_region`)
   - `background_mask = 1 - object_mask`
   - `plane_mask` via quantil de `depth_hint`.
4. **Inferência de profundidade relativa**:
   - `depth_hint` (assimetria angular + distância da borda)
   - `depth_order_map` normalizado.
5. **Lifting para pseudo-volume**:
   - `volume_lift` em bins de profundidade
   - projeção média do volume como canal de ocupação.
6. **Pistas de superfície e angularidade**:
   - `surface_hint` por gradiente de depth (dx, dy)
   - `angular_features` com `angular_perspective_matrix`.

## Canais para a CNN
A entrada aumentada usa **9 canais**:
- canal 0: imagem original em grayscale
- canais 1..8: edge, plane_mask, object_mask, depth_hint, depth_order, occupancy_projection, surface_dx, surface_dy

## Experimento MNIST
Script: `experiments/mnist_foldpy_experiment.py`

Compara:
- **Baseline:** CNN pequena com 1 canal (imagem bruta)
- **FoldPy:** mesma CNN com 9 canais (imagem + estrutura)

Métricas salvas em JSON:
- `train_loss`
- `test_accuracy`
- `train_seconds`
- `inference_ms_per_batch`
- deltas baseline vs FoldPy

## Execução
```bash
python experiments/mnist_foldpy_experiment.py --epochs 2 --subset 20000
```

## Limitações
- Não há geometria física real nem calibração de câmera.
- `object_mask` é limiar simples, podendo falhar em imagens complexas.
- `plane_mask` é heurística de quantil.
- O pseudo-volume representa ordenação/ocupação aproximada, não malha 3D.
- Em MNIST, ganhos podem ser modestos e dependentes de hiperparâmetros.
