"""Geometry constraints for moving arms against non-adjacent robot bodies."""
def arm_collision_pairs(model):
    geoms=[g for g in range(model.ngeom)
        if model.body(int(model.geom_bodyid[g])).name.startswith('robot/')
        and (model.geom_contype[g] or model.geom_conaffinity[g])]
    exclusions=set(int(x) for x in model.exclude_signature)
    pairs=[]
    for i,a in enumerate(geoms):
        for b in geoms[i+1:]:
            ba,bb=int(model.geom_bodyid[a]),int(model.geom_bodyid[b])
            na,nb=model.body(ba).name,model.body(bb).name
            if not any(('Arm_' in n or 'hand_' in n) for n in (na,nb)): continue
            if not ((int(model.geom_contype[a])&int(model.geom_conaffinity[b])) or
                    (int(model.geom_contype[b])&int(model.geom_conaffinity[a]))): continue
            wa,wb=int(model.body_weldid[ba]),int(model.body_weldid[bb])
            if wa==wb: continue
            pa=int(model.body_weldid[int(model.body_parentid[wa])])
            pb=int(model.body_weldid[int(model.body_parentid[wb])])
            if wa==pb or wb==pa: continue
            low,high=sorted((ba,bb)); wl,wh=sorted((wa,wb))
            if ((low<<16)|high) in exclusions or ((wl<<16)|wh) in exclusions: continue
            pairs.append((a,b))
    return pairs
