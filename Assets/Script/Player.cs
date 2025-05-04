using Unity.VisualScripting;
using UnityEngine;

public class Player : MonoBehaviour
{
    [SerializeField]
    protected float m_Speed;
    protected BoxCollider2D m_Collider;
    protected Rigidbody2D m_RigidBody;
    [SerializeField]
    protected bool canMove;
    protected bool m_IsSafe;
    protected GameObject StageManager;

    void Start()
    {
        canMove = true;
        m_IsSafe = false;
        if (null == m_Collider)
        {
            m_Collider = GetComponent<BoxCollider2D>();
        }
        if (null == m_RigidBody)
        {
            m_RigidBody = GetComponent<Rigidbody2D>();
            //m_RigidBody.layerMask = LayerMask.GetMask("Wall");
        }
        if (null == StageManager)
        {
            StageManager = GameObject.Find("StageManager");
        }
    }
    private void Move()
    {
        if (Input.anyKey)
        {
            float halfSize = (transform.lossyScale.x + transform.lossyScale.y) * 0.6f * 0.6f;
            float rayAdjust = halfSize - 0.05f;

            float xDir = Input.GetAxisRaw("Horizontal");
            float yDir = Input.GetAxisRaw("Vertical");

            Vector2 xVec = new Vector2(xDir, 0f);
            Vector2 yVec = new Vector2(0f, yDir);

            int layer = 1 << LayerMask.NameToLayer("Wall");

            Vector2 pos = transform.position;

            RaycastHit2D[] raycastHits = new RaycastHit2D[4];

            float xMove = m_Speed * Time.deltaTime;
            float yMove = m_Speed * Time.deltaTime;

            raycastHits[0] = Physics2D.Raycast(pos + yVec * halfSize + (Vector2.right * rayAdjust), yVec, yMove, layer);
            raycastHits[1] = Physics2D.Raycast(pos + yVec * halfSize + (Vector2.left * rayAdjust), yVec, yMove, layer);
            raycastHits[2] = Physics2D.Raycast(pos + xVec * halfSize + (Vector2.up * rayAdjust), xVec, xMove, layer);
            raycastHits[3] = Physics2D.Raycast(pos + xVec * halfSize + (Vector2.down * rayAdjust), xVec, xMove, layer);

            // 상, 하
            if (raycastHits[0].transform != null)
            {
                if (raycastHits[0].distance <= yMove)
                {
                    yMove = raycastHits[0].distance;
                }
            }
            if (raycastHits[1].transform != null)
            {
                if (raycastHits[1].distance <= yMove)
                {
                    yMove = raycastHits[1].distance;
                }
            }
            if (raycastHits[2].transform != null)
            {
                if (raycastHits[2].distance <= xMove)
                {
                    xMove = raycastHits[2].distance;
                }
            }
            if (raycastHits[3].transform != null)
            {
                if (raycastHits[3].distance <= xMove)
                {
                    xMove = raycastHits[3].distance;
                }
            }

            Vector2 temp = new Vector2(xDir * xMove, yDir * yMove);

            transform.Translate(temp);
        }

    }
    protected void FixedUpdate()
    {
        Move();
        // if (!canMove)
        // {
        //     Vector2 force = Vector2.zero;

        //     force += Vector2.right * Input.GetAxisRaw("Horizontal") * m_Speed;
        //     force += Vector2.up * Input.GetAxisRaw("Vertical") * m_Speed;

        //     m_RigidBody.AddForce(force);
        // }
    }

    void OnTriggerEnter2D(Collider2D collider)
    {
        switch (collider.tag)
        {
            case "Enemy":
                Debug.Log("Enemy");
                TriggerEnterEnemy(collider);
                break;
            case "SafetyZone":
                TriggerSafetyZone(collider);
                break;
            case "Coin":
                TrigerEnterCoin(collider);
                break;
        }
    }
    private void TriggerEnterEnemy(Collider2D collider)
    {
        //if (M_Edit.isEditMode)
        //return;

        if (!m_IsSafe)
        {
            StageManager.GetComponent<StageManager>().Death();
        }
    }
    private void TriggerSafetyZone(Collider2D collider)
    {
        m_IsSafe = true;
        StageManager.GetComponent<StageManager>().EnterSafetyZone();
    }

    void TrigerEnterCoin(Collider2D collider)
    {
        collider.gameObject.SetActive(false);
        StageManager.GetComponent<StageManager>().EnterCoin(collider);
    }
}